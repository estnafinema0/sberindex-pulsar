"""Конвейерный драйвер интерпретации: «паспорта типов» для отчёта и лендинга.

Запуск: `python -m pulsar.run_interpret` (метод/K/опорный месяц — из configs/final.yaml);
отладка на другом разбиении: `python -m pulsar.run_interpret --method kmeans --k 4`.

Метки: предпочтительно outputs/dynamics/{labels_confirmed,labels_raw}.npy (T×N, сквозные номера, пишет run_dynamics);
если их нет (или запрошен метод/K не из final.yaml) — outputs/cluster/labels_{method}_K{k}.npy, выровненные
align_sequence, а подтверждённые метки считаются тут же правилом dynamics.confirmed_transitions (min_run).

Тип МО для паспорта — модальный подтверждённый тип за период (по умолчанию 2024 год, `--period all` — все месяцы);
признаки усредняются по тем же месяцам. Отдельно — профиль опорного месяца reference_month (сырые метки месяца).
Внешние переменные (Росстат 2023, доступность рынков, тип/статус/регион МО) в кластеризацию НЕ входили —
это внешняя проверка.

Выход: outputs/interpret/{mirkin_profile.csv, readable_profile.csv, mirkin_profile_ref.csv, readable_profile_ref.csv,
external_validation.csv, rule_tree.txt, rule_tree.json, typical_border.csv, membership_margin.npy,
type_names.json, shap_importance.csv, types_summary.json, mo_table.parquet}.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import warnings

import numpy as np
import pandas as pd

from pulsar import align, data, dynamics, features, interpret, run_cluster

ROOT = data.ROOT
OUT = ROOT / "outputs" / "interpret"
DYN = ROOT / "outputs" / "dynamics"
LEVEL_NAME = "Итог на жителя"   # «сырое» имя уровня: _split_raw_names узнаёт его по слову «итог»

# Росстат 2023: какие колонки идут во внешнюю проверку. Абсолютные численности по отраслям (emp_*, wage_*)
# не берём — у них 20–85 % пропусков и они дублируют доли; служебные колонки сопоставления тоже не нужны.
ROSSTAT_NUMERIC = ["population", "population_urban_share", "wage_total", "wage_rel", "employees_total",
                   "emp_per_capita", "hhi_employment", "extractive_share", "public_share", "residual_share"]
ROSSTAT_CATEGORICAL = ["top_sector"]


# ----------------------------------------------------------------------------- метки
def _final_cfg() -> dict:
    return data.load_config("final")


def load_label_cubes(method: str, k: int, source: str = "auto") -> tuple[np.ndarray, np.ndarray, str]:
    """(L_conf, L_raw, источник). L_* — (T, N) сквозные метки.

    source: auto — dynamics, если файлы есть и метод/K совпадают с final.yaml, иначе cluster;
    dynamics / cluster — принудительно."""
    fin = _final_cfg()
    is_final = (method == fin["method"] and int(k) == int(fin["k"]))
    conf_p, raw_p = DYN / "labels_confirmed.npy", DYN / "labels_raw.npy"
    use_dyn = source == "dynamics" or (source == "auto" and is_final and conf_p.exists())
    if use_dyn:
        if not conf_p.exists():
            raise FileNotFoundError(f"нет {conf_p} — сначала run_dynamics или --source cluster")
        L_conf = np.load(conf_p).astype(np.int64)
        L_raw = np.load(raw_p).astype(np.int64) if raw_p.exists() else L_conf.copy()
        return L_conf, L_raw, f"dynamics ({conf_p.name})"
    # запасной путь: помесячные метки → сквозные номера → подтверждение по правилу min_run
    L = run_cluster.load_labels(method, k)
    L_raw = np.vstack(align.align_sequence(list(L))).astype(np.int64)
    min_run = int(data.load_config("dynamics").get("min_run", 3))
    L_conf, _ = dynamics.confirmed_transitions(L_raw, min_run=min_run)
    return L_conf, L_raw, f"cluster (labels_{method}_K{k}.npy + align_sequence + min_run={min_run})"


def modal_labels(L: np.ndarray, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Модальный тип по месяцам sel (отрицательные метки = «не определён» — пропускаются).
    Ничья решается в пользу типа последнего месяца периода. Возвращает (мода, доля месяцев в моде)."""
    S = L[sel]
    vals = np.unique(S[S >= 0])
    counts = np.stack([(S == v).sum(axis=0) for v in vals], axis=1).astype(float)   # (N, K)
    last = S[-1]
    counts += 0.5 * (last[:, None] == vals[None, :])                                 # тай-брейк, не меняет строгий максимум
    j = counts.argmax(axis=1)
    mode = vals[j]
    share = np.floor(counts[np.arange(len(j)), j]) / len(sel)
    return mode.astype(np.int64), share


# ----------------------------------------------------------------------------- «сырые» признаки
def raw_cubes(ids: np.ndarray, months: list[str]) -> dict:
    """Доли шести категорий (T, N, 6), итог на жителя ₽ (T, N) и итог / медиана страны месяца (T, N).
    Медиана — по всем МО месяца (как в features.raw_features), не только по панели."""
    fcfg = data.load_config("features")
    w = features.wide_table(data.read_consumption(), fcfg)
    comps = fcfg["categories"] + [fcfg["other_name"]]
    total = fcfg["total"]
    tot_all = w[total].unstack("date").reindex(columns=months).astype(float)
    nat_med = tot_all.median(axis=0)                                  # медиана страны в каждом месяце
    tot = tot_all.reindex(index=ids)
    shares = np.stack([(w[c].unstack("date").reindex(index=ids, columns=months).astype(float) / tot).to_numpy().T
                       for c in comps], axis=2)
    return {"comps": comps, "shares": shares, "total": tot.to_numpy().T, "rel": (tot / nat_med).to_numpy().T}


def raw_matrix(cubes: dict, sel: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Средние за месяцы sel: доли (0..1) и итог ₽ → матрица (N, 7) для readable_profile / rule_tree."""
    sh = np.nanmean(cubes["shares"][sel], axis=0)
    lv = np.nanmean(cubes["total"][sel], axis=0)
    return np.column_stack([sh, lv]), cubes["comps"] + [LEVEL_NAME]


# ----------------------------------------------------------------------------- имена типов
def _names_file_payload(names_dict: dict, problems: list[str], n_by_type: dict, ctx: dict) -> dict:
    """type_names.json в виде, удобном для ручной правки: список типов с id/name/short/description."""
    return {
        "_как_править": ("Правьте name/short/description у нужных типов и поставьте locked=true: тогда run_interpret "
                         "возьмёт ваши имена (и проверит их check_names), а не черновые. Поле rule — правило "
                         "configs/interpret.yaml, по которому проверяется имя; null — имя не проверяется кодом."),
        "locked": False, **ctx,
        "types": [{"id": int(k), "name": v["name"], "short": v["short"], "description": v["description"],
                   "rule": v.get("rule"), "n": int(n_by_type.get(k, 0)), "top_features": list(v.get("top_features", []))}
                  for k, v in names_dict.items()],
        "check_names": problems,
    }


def load_locked_names(path: pathlib.Path, ctx: dict) -> dict | None:
    """Имена, поправленные человеком (locked=true и тот же метод/K/период), иначе None."""
    if not path.exists():
        return None
    try:
        js = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        warnings.warn(f"{path.name} не читается ({e}) — беру черновые имена")
        return None
    if not js.get("locked") or any(js.get(key) != val for key, val in ctx.items()):
        return None
    return {int(t["id"]): {"name": t["name"], "short": t.get("short", t["name"]), "description": t.get("description", ""),
                           "rule": t.get("rule"), "top_features": t.get("top_features", [])} for t in js["types"]}


# ----------------------------------------------------------------------------- сводка
def _rename_summary(summary: dict, code2ru: dict) -> dict:
    """Ключи признаков в профиле/SHAP/топ-признаках сводки → русские подписи (как readable_feature_names)."""
    for t in summary["types"]:
        t["profile"] = {code2ru.get(f, f): v for f, v in t.get("profile", {}).items()}
        if "shap" in t:
            t["shap"] = {code2ru.get(f, f): v for f, v in t["shap"].items()}
        for r in t.get("top_features", []):
            r["label"] = code2ru.get(r["feature"], r["feature"])
    return summary


def _fmt_table(rows: list[list[str]], head: list[str]) -> str:
    w = [max(len(str(r[i])) for r in rows + [head]) for i in range(len(head))]
    line = lambda r: "  ".join(str(c).ljust(w[i]) for i, c in enumerate(r))  # noqa: E731
    return "\n".join([line(head), line(["─" * x for x in w])] + [line(r) for r in rows])


# ----------------------------------------------------------------------------- конвейер
def run(method: str | None = None, k: int | None = None, period: str = "2024", source: str = "auto",
        reference_month: str | None = None, out: pathlib.Path = OUT, do_shap: bool = True) -> dict:
    t0 = time.time()
    fin = _final_cfg()
    method = method or fin["method"]
    k = int(k or fin["k"])
    ref_month = reference_month or str(fin.get("reference_month"))
    cfg = interpret.load_config()
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # --- признаки и метки
    f = features.load()
    X, names, ids, months, panel = f["X"].astype(float), f["names"], f["ids"], f["months"], f["panel"]
    L_conf, L_raw, src = load_label_cubes(method, k, source)
    if L_conf.shape != X.shape[:2]:
        raise ValueError(f"метки {L_conf.shape} не совпадают с признаками {X.shape[:2]}")
    sel = np.arange(len(months)) if period == "all" else np.array([t for t, m in enumerate(months) if m.startswith(period)])
    if sel.size == 0:
        raise ValueError(f"нет месяцев для периода {period!r}")
    t_ref = months.index(ref_month)
    labels, mode_share = modal_labels(L_conf, sel)
    labels_ref = L_raw[t_ref]                       # опорный месяц — фактическое разбиение этого месяца
    Xp = X[sel].mean(axis=0)                        # стандартизованные признаки, усреднённые по периоду

    meta = panel.set_index("territory_id").reindex(ids).reset_index()
    meta["name"] = meta["municipal_district_name_short"]
    meta["municipal_district_status"] = meta["municipal_district_status"].fillna("нет статуса")  # NaN = обычное МО

    cubes = raw_cubes(ids, months)
    lv_chk = np.nanmax(np.abs(cubes["total"] - f["levels"]))
    if lv_chk > 1e-6:
        warnings.warn(f"итог из consumption расходится с features.levels (max |Δ| = {lv_chk:.3g})")
    raw, raw_names = raw_matrix(cubes, sel)
    raw_ref, _ = raw_matrix(cubes, np.array([t_ref]))
    ru = interpret.readable_feature_names(raw_names, [LEVEL_NAME], cfg)
    code2ru = dict(zip(names, ru))                  # clr_<кат> → «Доля «<кат>», %», log_level_rel → «Итог на жителя, ₽»
    raw2ru = {f"{n}{cfg['readable']['share_suffix']}": r for n, r in zip(raw_names[:-1], ru[:-1])}

    # --- 1–2. профиль по Миркину и паспорт в исходных единицах (период и опорный месяц)
    prof = interpret.mirkin_profile(Xp, labels, names)
    readable = interpret.readable_profile(raw, labels, raw_names, [LEVEL_NAME], cfg)
    prof_ref = interpret.mirkin_profile(X[t_ref], labels_ref, names)
    readable_ref = interpret.readable_profile(raw_ref, labels_ref, raw_names, [LEVEL_NAME], cfg)
    prof.to_csv(out / "mirkin_profile.csv", index=False)
    prof_ref.to_csv(out / "mirkin_profile_ref.csv", index=False)
    readable.rename(columns=raw2ru).to_csv(out / "readable_profile.csv")
    readable_ref.rename(columns=raw2ru).to_csv(out / "readable_profile_ref.csv")

    # --- 6. имена: черновик name_types или поправленные человеком (locked) + проверка кодом
    n_by_type = pd.Series(labels).value_counts().to_dict()
    ctx = {"method": method, "k": k, "period": period}
    names_path = out / "type_names.json"
    # приоритет — имена из репозитория (configs/type_names.json, правятся человеком и переживают make clean), затем outputs/
    locked = load_locked_names(data.ROOT / "configs" / "type_names.json", ctx) or load_locked_names(names_path, ctx)
    names_dict = locked if locked is not None else interpret.name_types(prof, readable, None, cfg)
    names_dict = {int(t): v for t, v in names_dict.items()}
    problems = interpret.check_names(prof, names_dict, cfg)
    if locked is None:                               # ручные правки не затираем
        names_path.write_text(json.dumps(interpret._to_py(_names_file_payload(names_dict, problems, n_by_type, ctx)),
                                         ensure_ascii=False, indent=1), encoding="utf-8")

    # --- 3. внешняя проверка
    ros = pd.read_parquet(ROOT / "outputs" / "external" / "rosstat_2023.parquet")
    emp_share = [c for c in ros.columns if c.startswith("emp_share_")]
    ext = ros[["territory_id"] + ROSSTAT_NUMERIC + emp_share + ROSSTAT_CATEGORICAL].copy()
    ma = data.read_market_access()[["territory_id", "market_access"]]
    ext = ext.merge(ma.assign(territory_id=ma.territory_id.astype(int)), on="territory_id", how="outer")
    lab_by_tid = pd.Series(labels, index=ids.astype(int))
    ev = interpret.external_validation(ext, lab_by_tid, meta, cfg)
    ev.to_csv(out / "external_validation.csv", index=False)

    # --- 4. дерево правил в процентах и рублях
    rcfg = cfg["rule_tree"]
    rules = interpret.rule_tree(raw, labels, raw_names, rcfg["max_depth"], rcfg["min_leaf"], rcfg["cv_folds"],
                                cfg.get("seed", 42), [LEVEL_NAME], names_dict, cfg)
    (out / "rule_tree.txt").write_text(rules["text"], encoding="utf-8")
    (out / "rule_tree.json").write_text(json.dumps(interpret._to_py(
        {k_: rules[k_] for k_ in ("text", "accuracy", "cv_accuracy", "feature_names", "class_names", "leaf_purity", "types")}
        | {"max_depth": rcfg["max_depth"], "min_leaf": rcfg["min_leaf"], "cv_folds": rcfg["cv_folds"]}),
        ensure_ascii=False, indent=1), encoding="utf-8")

    # --- 5. типичные / пограничные МО и зазор принадлежности
    tb = interpret.typical_and_border(Xp, labels, meta, cfg["typical"]["n"])
    tb.to_csv(out / "typical_border.csv", index=False)
    margin = interpret.membership_margin(Xp, labels)
    np.save(out / "membership_margin.npy", margin)

    # --- 7. SHAP (если lightgbm/shap доступны)
    shap_df = interpret.shap_importance(Xp, labels, names, cfg.get("seed", 42)) if do_shap else None
    if shap_df is not None:
        shap_df.to_csv(out / "shap_importance.csv")

    # --- 8. сводка для лендинга
    summary = interpret.type_summary(labels, meta, prof, readable.rename(columns=raw2ru), tb, names_dict, ev,
                                     rules, shap_df, cfg["mirkin"]["top_n"])
    summary = _rename_summary(summary, code2ru)
    agree = float((labels == labels_ref).mean())
    summary.update(interpret._to_py({
        "method": method, "k_config": k, "period": period, "months": [months[t] for t in sel], "labels_source": src,
        "feature_labels": code2ru, "check_names": problems, "names_locked": locked is not None,
        "reference_month": {"month": ref_month, "n_by_type": pd.Series(labels_ref).value_counts().sort_index().to_dict(),
                            "B_over_T": prof_ref.attrs["B_over_T"], "share_same_as_mode": agree,
                            "medians": readable_ref.rename(columns=raw2ru).to_dict(orient="index")},
    }))
    (out / "types_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- таблица МО для карты лендинга
    tag = period
    mo = pd.DataFrame({"territory_id": ids.astype(int), "name": meta["name"], "name_full": meta["municipal_district_name"],
                       "region": meta["region_name"], "mo_type": meta["municipal_district_type"],
                       f"type_mode_{tag}": labels, "type_short": [names_dict[int(t)]["short"] for t in labels],
                       "mode_share": mode_share, "type_ref_month": labels_ref, "margin": margin,
                       "n_confirmed_changes": dynamics.n_changes(L_conf), "n_raw_changes": dynamics.n_changes(L_raw),
                       "lat": meta["lat"], "lon": meta["lon"],
                       f"level_{tag}": np.nanmean(cubes["total"][sel], axis=0),
                       f"level_rel_{tag}": np.nanmean(cubes["rel"][sel], axis=0)})
    for j, c in enumerate(cubes["comps"]):
        mo[f"share_{c}_{tag}"] = raw[:, j]
    mo.to_parquet(out / "mo_table.parquet", index=False)

    # --- консоль
    wage = ext.set_index("territory_id")["wage_total"].reindex(ids.astype(int)).to_numpy()
    rows = []
    for t in sorted(n_by_type):
        m = labels == t
        tf = interpret.top_features(prof, t, 3)
        top = ", ".join(f"{cfg['naming']['feature_labels'].get(fn, fn)}{s}" for fn, s in zip(tf["feature"], tf["sign"]))
        typ = tb[(tb["type"] == t) & (tb["role"] == "typical")]["name"].tolist()
        rows.append([t, int(m.sum()), f"{m.mean():.1%}", names_dict[int(t)]["name"], top,
                     f"{np.nanmedian(raw[m, -1]):,.0f}".replace(",", " "), f"{np.nanmedian(wage[m]):,.0f}".replace(",", " "),
                     "; ".join(typ)])
    print(f"\nИнтерпретация: {method} K={k}, период {period} ({len(sel)} мес.), метки: {src}")
    print(_fmt_table(rows, ["тип", "n", "доля", "имя (черновик)" if locked is None else "имя", "топ-3 (Миркин)",
                            "траты ₽/жит.", "зарплата ₽", "типичные МО"]))
    print(f"B/T = {prof.attrs['B_over_T']:.3f} (период), {prof_ref.attrs['B_over_T']:.3f} ({ref_month}); "
          f"тип {ref_month} = модальный у {agree:.1%} МО; дерево: точность {rules['accuracy']:.3f}, CV {rules['cv_accuracy']:.3f}")
    top_ext = ev.head(6)
    print("внешняя проверка (сильнейшие): " + "; ".join(
        f"{r.variable} {'η²_H' if r.kind == 'numeric' else 'V'}={r.strength:.2f}" for r in top_ext.itertuples()))
    print("check_names: " + ("все имена согласуются с профилями" if not problems else ""))
    for p in problems:
        print("  ВНИМАНИЕ:", p)
    print(f"готово за {time.time() - t0:.1f} с → {out}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Паспорта типов МО (интерпретация итоговой типологии)")
    ap.add_argument("--method", default=None, help="метод (по умолчанию configs/final.yaml)")
    ap.add_argument("--k", type=int, default=None, help="K (по умолчанию configs/final.yaml)")
    ap.add_argument("--period", default="2024", help="год 'YYYY' или 'all' — месяцы для модального типа и средних")
    ap.add_argument("--source", default="auto", choices=["auto", "dynamics", "cluster"], help="откуда брать метки")
    ap.add_argument("--reference-month", default=None, help="опорный месяц (по умолчанию из final.yaml)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-shap", action="store_true", help="пропустить LightGBM+SHAP")
    a = ap.parse_args()
    run(a.method, a.k, a.period, a.source, a.reference_month, pathlib.Path(a.out), not a.no_shap)


if __name__ == "__main__":
    main()
