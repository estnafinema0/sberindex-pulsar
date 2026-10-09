"""Экспорт данных для лендинга по контракту docs/site_data_contract.md. `python -m pulsar.site`.

Собирает outputs/{features,dynamics,interpret,compare,site,external} → site/data/*.json и копирует геометрию.
Все числа берутся из артефактов конвейера — ничего не вычисляется заново, кроме агрегатов для отображения.
"""
from __future__ import annotations

import datetime as dt
import re
import json
import pathlib
import shutil

import numpy as np
import pandas as pd

from pulsar import data, features

OUTS = data.ROOT / "outputs"
SITE = data.ROOT / "site" / "data"
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#999999", "#8B4513", "#2F4F4F", "#F0E442"]  # Okabe–Ito без жёлтого (как в site/js/app.js)


def _py(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return [_py(v) for v in o.tolist()]
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def dump(name: str, obj) -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / name).write_text(json.dumps(_py(obj), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _read_json(path: pathlib.Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def build() -> None:
    final = data.load_config("final")
    f = features.load()
    months, ids, ids_all = f["months"], f["ids"], f["ids_all"]
    panel = f["panel"].set_index("territory_id")
    T, N = len(months), len(ids)
    dyn = OUTS / "dynamics"
    L_raw = np.load(dyn / "labels_raw.npy"); L_conf = np.load(dyn / "labels_confirmed.npy")
    conf = np.load(dyn / "confidence.npy") if (dyn / "confidence.npy").exists() else np.ones_like(L_raw, dtype=float)
    K = int(L_conf.max()) + 1
    L_raw = np.where(L_raw >= K, -1, L_raw)  # сырые метки сверх подтверждённых типов (артефакт выравнивания) — «вне типов»
    summary = _read_json(dyn / "summary.json", {})
    oot = _read_json(dyn / "out_of_time.json", {})

    # --- meta
    dump("meta.json", {"months": months, "k": int(final["k"]), "k_effective": K, "method": final["method"], "layer": final["layer"],
                       "generated": dt.date.today().isoformat(), "n_panel": int(N), "n_all": int(len(ids_all)),
                       "attribution": "Данные: Лаборатория СберИндекс, CC BY-SA 4.0"})

    # --- types (имена — из type_names.json, их правит человек; остальное — из types_summary.json)
    interp = OUTS / "interpret"
    tn = _read_json(data.ROOT / "configs" / "type_names.json", {})
    names = {int(d["id"]): d for d in tn.get("types", [])} if isinstance(tn, dict) else {}
    ts = _read_json(interp / "types_summary.json", {})
    info = {int(d["type"]): d for d in ts.get("types", [])} if isinstance(ts, dict) else {}
    medians_all = ts.get("medians_all", {}) if isinstance(ts, dict) else {}
    mode24 = pd.DataFrame(L_conf[12:]).mode(axis=0).iloc[0].to_numpy().astype(int) if T >= 24 else L_conf[-1]
    mt = interp / "mo_table.parquet"
    mo_tab = pd.read_parquet(mt) if mt.exists() else None
    if mo_tab is not None:  # тип МО = модальный подтверждённый тип 2024 (как в отчёте)
        counts = np.bincount(mo_tab["type_mode_2024"].dropna().astype(int), minlength=K)[:K]
        mode_by_id = dict(zip(mo_tab["territory_id"].astype(int), mo_tab["type_mode_2024"].astype(int)))
    else:
        counts = np.bincount(mode24, minlength=K); mode_by_id = {}
    N_types = int(counts.sum()) or N
    EXT_RU = {"population": "Население, чел.", "population_urban_share": "Доля городского населения", "wage_total": "Средняя зарплата, ₽",
              "employees_total": "Работников организаций", "market_access": "Доступность рынков", "extractive_share": "Доля добычи и энергетики в занятости",
              "public_share": "Доля бюджетного сектора", "hhi_employment": "Концентрация занятости (HHI)", "top_sector": "Ведущая отрасль"}
    def ext_fmt(key, v):
        if not isinstance(v, (int, float)): return str(v)
        if key.endswith("_share"): return f"{100 * v:.1f}".replace(".", ",") + " %"
        if key == "hhi_employment": return f"{v:.2f}".replace(".", ",")
        return f"{v:,.0f}".replace(",", "\u00a0")
    short_feat = lambda lab: lab.replace("Доля «", "").replace("», %", "").replace("Итог на жителя, % к медиане", "Уровень трат")
    types = []
    for k in range(K):
        nm, inf = names.get(k, {}), info.get(k, {})
        prof = {short_feat(lab): round(float(v["mean"]), 3) for lab, v in inf.get("profile", {}).items()} if inf.get("profile") else {}
        meds = {lab: (round(float(v), 1) if isinstance(v, (int, float)) else v) for lab, v in inf.get("medians", {}).items() if lab != "n"}
        em = inf.get("external_medians") or {}
        ext = {EXT_RU[key]: ext_fmt(key, em[key]) for key in EXT_RU if key in em and em[key] is not None}
        pick = lambda lst: [{"territory_id": int(d["territory_id"]), "name": d["name"], "region": d.get("region_name", "")} for d in lst[:3]]
        types.append({"id": k, "name": nm.get("name", inf.get("name", f"Тип {k + 1}")), "short": nm.get("short", inf.get("short", f"Т{k + 1}")),
                      "color": PALETTE[k % len(PALETTE)], "description": nm.get("description", inf.get("description", "")),
                      "n": int(counts[k]), "share": float(counts[k] / N_types), "profile": prof, "medians": meds, "medians_all": {lab: v for lab, v in medians_all.items() if lab != "n"},
                      "typical": pick(inf.get("typical", [])), "border": pick(inf.get("border", [])), "external": ext,
                      "top_features": [d.get("label", d.get("feature")) for d in inf.get("top_features", [])][:3]})
    dump("types.json", types)

    # --- labels per MO (панель + вне панели по ближайшему типу, если есть mo_table)
    cons = data.read_consumption()
    cfg_f = data.load_config("features")
    cats = cfg_f["categories"]; total = cfg_f["total"]
    w = cons.pivot_table(index=["territory_id", "date"], columns="category", values="value", aggfunc="first")
    labels = {}
    idx = {tid: j for j, tid in enumerate(ids)}
    pos_all = {tid: j for j, tid in enumerate(ids_all)}
    X_all = f["X_all"].astype(float); X = f["X"].astype(float)
    centers = [np.vstack([X[t][L_conf[t] == k].mean(axis=0) if (L_conf[t] == k).any() else np.full(X.shape[2], np.inf) for k in range(K)]) for t in range(T)]
    for tid in ids_all:
        row = panel.loc[tid]
        rec = {"name": row["municipal_district_name_short"], "region": row["region_name"], "mo_type": row["municipal_district_type"]}
        try:
            sub = w.loc[tid].reindex(months)
            rec["level"] = [None if np.isnan(v) else int(v) for v in sub[total].to_numpy(float)]
            rec["shares"] = {c: [None if np.isnan(v) else round(float(v), 4) for v in (sub[c] / sub[total]).to_numpy(float)] for c in cats}
        except KeyError:
            rec["level"], rec["shares"] = [None] * T, {c: [None] * T for c in cats}
        if tid in idx:
            j = idx[tid]
            rec["type"] = L_raw[:, j].astype(int).tolist()
            rec["confirmed"] = L_conf[:, j].astype(int).tolist()
            rec["confidence"] = [round(float(v), 3) for v in conf[:, j]]
            rec["in_panel"] = True
        if int(tid) in mode_by_id:
            rec["type_2024"] = mode_by_id[int(tid)]
        else:  # вне панели: ближайший центр подтверждённого типа в каждом месяце; уверенность = зазор до второго центра
            j = pos_all[tid]
            typ, cf = [], []
            for t in range(T):
                x = X_all[t, j]
                if np.isnan(x).any():
                    typ.append(-1); cf.append(0.0); continue
                d = ((centers[t] - x) ** 2).sum(axis=1)
                o = np.argsort(d)
                typ.append(int(o[0])); cf.append(round(float(min(1.0, (d[o[1]] - d[o[0]]) / max(d[o[0]], 1e-9))), 3))
            rec["type"] = typ; rec["confirmed"] = typ; rec["confidence"] = cf; rec["in_panel"] = False
        labels[str(int(tid))] = rec
    dump("labels.json", labels)

    # --- transitions
    tr = {"shorrocks": {}, "stable_share": summary.get("stable_share_confirmed", summary.get("stable_share")), "events": [], "sizes": {}, "flows": []}
    m1 = dyn / "transition_matrix_lag1_confirmed.csv"
    tr["matrix"] = pd.read_csv(m1, index_col=0).fillna(0).to_numpy().tolist() if m1.exists() else []
    big = dyn / "transitions_2023-12_to_2024-12.csv"
    if big.exists():
        b = pd.read_csv(big)
        cols = {c.lower(): c for c in b.columns}
        fr, to, n = cols.get("from", list(b.columns)[0]), cols.get("to", list(b.columns)[1]), cols.get("count", cols.get("n", list(b.columns)[2]))
        tr["flows"] = [{"from": int(r[fr]), "to": int(r[to]), "n": int(r[n]), "period": "2023-12 → 2024-12"} for _, r in b.iterrows() if int(r[n]) > 0]
        M = b.pivot_table(index=fr, columns=to, values=n, aggfunc="sum").reindex(index=range(K), columns=range(K)).fillna(0)
        tr["counts"] = M.to_numpy().astype(int).tolist()
        tr["matrix_2023_2024"] = (M.div(M.sum(axis=1).replace(0, 1), axis=0)).round(4).to_numpy().tolist()
    sh = summary.get("shorrocks", {})
    g = lambda kind, lag: (sh.get(kind, {}) or {}).get(lag) if isinstance(sh.get(kind), dict) else None
    tr["shorrocks"] = {"raw": g("raw", "lag1"), "confirmed": g("confirmed", "lag1"), "raw_lag12": g("raw", "lag12"), "confirmed_lag12": g("confirmed", "lag12"),
                       "confirmed_2023_12_to_2024_12": summary.get("shorrocks_2023_12_to_2024_12_confirmed")}
    ss = summary.get("stable_share", {})
    tr["stable_share"] = ss.get("confirmed") if isinstance(ss, dict) else ss
    tr["stable_share_raw"] = ss.get("raw") if isinstance(ss, dict) else None
    tr["n_changes"] = summary.get("n_changes", {})
    sz = dyn / "sizes_over_time.csv"
    if sz.exists():
        s = pd.read_csv(sz, index_col=0)
        s.columns = [int(c) for c in s.columns]; s.index = s.index.astype(str)
        tr["sizes"] = {str(m): [int(v) for v in s.loc[m].reindex(range(K), fill_value=0)] if m in s.index else [] for m in months}
    ge = dyn / "greene_events.csv"
    if ge.exists():
        g = pd.read_csv(ge)
        if "labels" in g.columns:
            g = g[g["labels"].astype(str).str.contains("conf")] if g["labels"].astype(str).str.contains("conf").any() else g
        ints = lambda v: [int(x) for x in re.findall(r"-?\d+", str(v))] if not isinstance(v, (int, np.integer)) else [int(v)]
        tr["events"] = [{"month": str(r.get("month")), "event": r.get("event"), "from": ints(r.get("from_ids", r.get("from", ""))),
                         "to": ints(r.get("to_ids", r.get("to", ""))), "jaccard": r.get("jaccard")} for _, r in g.head(200).iterrows()]
    # ряды для графиков кейсов без конкретного МО
    tr["changes_by_month"] = summary.get("changes", {}).get("by_month", {})
    try:
        mo_tab = pd.read_parquet(OUTS / "interpret" / "mo_table.parquet")[["territory_id", "type_mode_2024"]]
        share = (w["Маркетплейсы"] / w[total]).unstack("date").reindex(mo_tab.territory_id.to_numpy())
        share["t"] = mo_tab.type_mode_2024.to_numpy()
        med = share.groupby("t")[months].median() * 100
        tr["marketplaces_by_type"] = {str(int(k)): [round(float(v), 2) for v in med.loc[k].to_numpy()] for k in med.index}
    except Exception as e:  # график кейса не должен ронять экспорт
        print("marketplaces_by_type:", e)
    tr["out_of_time"] = oot
    tr["summary"] = summary
    dump("transitions.json", tr)

    # --- methods
    comp = OUTS / "compare"
    meth = {"k_fixed": int(final["k"]), "rows": [], "pareto": [], "layers": [], "notes": "AVU вырожден при K≤3 (1 при K=2, 2/3 при K=3); MQ = модулярность Q; TurboMQ ≡ K·AVI."}
    rk = comp / "ranking_fixed_k.csv"
    if rk.exists():
        r = pd.read_csv(rk, index_col=0)
        for mname, row in r.iterrows():
            meth["rows"].append({"method": mname, **{c: row[c] for c in r.columns}})
            meth["pareto"].append({"method": mname, "SW": row.get("SW"), "MQ": row.get("MQ")})
    ab = comp / "ablation_layers.csv"
    if ab.exists():
        a = pd.read_csv(ab)
        meth["layers"] = a.to_dict("records")
    ari = comp / "ablation_ari.csv"
    if ari.exists():
        meth["layers_ari"] = pd.read_csv(ari).to_dict("records")
    pb = comp / "paired_bootstrap.csv"
    if pb.exists():
        meth["paired"] = pd.read_csv(pb).to_dict("records")
    dump("methods.json", meth)

    # --- cases (текст пишется человеком в docs/cases.json; здесь — копия)
    cases = _read_json(data.ROOT / "docs" / "cases.json", [])
    dump("cases.json", cases)

    # --- geometry
    for fn in ("mo.geojson", "regions.geojson", "mo_cities.geojson", "centroids.csv"):
        src = OUTS / "site" / fn
        if src.exists():
            shutil.copy(src, SITE / fn)
    print(f"→ {SITE}: {len(labels)} МО, {K} типов, {len(tr.get('flows', []))} потоков, методов {len(meth['rows'])}")


if __name__ == "__main__":
    build()
