"""Интерпретация типов МО: профили по Миркину, паспорт типа, внешняя проверка, дерево правил, имена.

Все функции чистые (массивы / DataFrame на входе), чтобы тестировать на синтетике.
Обозначения: X (N×F) — стандартизованные признаки, raw (N×F') — те же признаки в исходных единицах
(доли 0..1 и итог в ₽ на жителя), labels (N,) — метки типов, meta — справочник МО по строкам X.

Внешние переменные (Росстат, тип/статус МО, доступность рынков) в кластеризацию НЕ входили —
они используются только для проверки «снаружи» (η²_H, V Крамера, NMI).

Запуск: `python -m pulsar.interpret --labels <labels.npy>` → outputs/interpret/{profile.csv, readable.csv,
typical_border.csv, rules.txt, summary.json}.
"""
from __future__ import annotations

import json
import pathlib
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.stats.contingency import association
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "interpret"

ALL_LABEL = "Все МО"


def load_config(path: pathlib.Path | str | None = None) -> dict:
    with open(path or ROOT / "configs" / "interpret.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------------- вспомогательное
def _labels(labels) -> tuple[np.ndarray, list]:
    """Метки → (индексы 0..K−1, список исходных значений типов в порядке сортировки)."""
    lab = np.asarray(pd.Series(labels).to_numpy())
    types = sorted(pd.unique(lab).tolist())
    idx = {t: i for i, t in enumerate(types)}
    return np.array([idx[t] for t in lab]), types


def _centers(X: np.ndarray, enc: np.ndarray, k: int) -> np.ndarray:
    return np.vstack([X[enc == j].mean(axis=0) for j in range(k)])


def _to_py(o: Any) -> Any:
    """Рекурсивно приводит numpy/pandas к JSON-сериализуемым типам."""
    if isinstance(o, dict):
        return {str(k): _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return _to_py(o.reset_index().to_dict(orient="records"))
    if isinstance(o, pd.Series):
        return _to_py(o.to_dict())
    if isinstance(o, np.ndarray):
        return _to_py(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ----------------------------------------------------------------------------- 1. профиль по Миркину
def mirkin_profile(X: np.ndarray, labels, names: list[str]) -> pd.DataFrame:
    """Таблица тип × признак по Миркину (Clustering: A Data Recovery Approach, 2012).

    Пифагорово разложение разброса центрированных данных T = B + W, где
    B = Σ_k Σ_v n_k c_kv² (c_kv — среднее кластера после центрирования на общее среднее m_v).
    Колонки: n, share, mean (c_kv в исходной шкале X), global_mean (m_v), rel_dev — (c_kv − m_v)/|m_v|,
    а при m_v ≈ 0 (стандартизованные признаки) — (c_kv − m_v)/σ_v, т.е. в единицах СКО;
    contrib = n_k (c_kv − m_v)², contrib_pct_T — доля от T (%), contrib_pct_B — доля от B (%).
    В attrs: T, B, W, B_over_T, cluster_contrib (B_k/T по типам), types.
    """
    X = np.asarray(X, float)
    enc, types = _labels(labels)
    k = len(types)
    m = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Y = X - m
    T = float((Y**2).sum())
    rows = []
    B = 0.0
    cluster_contrib = {}
    for j, t in enumerate(types):
        mask = enc == j
        n_k = int(mask.sum())
        c = X[mask].mean(axis=0)
        dev = c - m
        contrib = n_k * dev**2
        B_k = float(contrib.sum())
        B += B_k
        cluster_contrib[t] = B_k / T if T > 0 else np.nan
        small = np.abs(m) < 1e-6 * sd
        rel = np.where(small, dev / sd, dev / np.where(small, 1.0, np.abs(m)))
        for v, name in enumerate(names):
            rows.append({"type": t, "feature": name, "n": n_k, "share": n_k / len(enc), "mean": float(c[v]),
                         "global_mean": float(m[v]), "rel_dev": float(rel[v]), "contrib": float(contrib[v])})
    prof = pd.DataFrame(rows)
    prof["contrib_pct_T"] = 100 * prof["contrib"] / T if T > 0 else np.nan
    prof["contrib_pct_B"] = 100 * prof["contrib"] / B if B > 0 else np.nan
    W = float(sum(((X[enc == j] - X[enc == j].mean(axis=0)) ** 2).sum() for j in range(k)))
    prof.attrs.update({"T": T, "B": B, "W": W, "B_over_T": B / T if T > 0 else np.nan,
                       "cluster_contrib": cluster_contrib, "types": types})
    return prof


def profile_wide(profile: pd.DataFrame, value: str = "mean") -> pd.DataFrame:
    """Профиль в широком виде: строки — типы, столбцы — признаки (для тепловой карты)."""
    return profile.pivot(index="type", columns="feature", values=value)


def top_features(profile: pd.DataFrame, k, n: int = 3) -> pd.DataFrame:
    """n признаков типа k с наибольшим вкладом n_k c_kv² (с знаком отклонения) — для подписи типа."""
    sub = profile[profile["type"] == k].sort_values("contrib", ascending=False).head(n)
    out = sub[["feature", "mean", "rel_dev", "contrib", "contrib_pct_T"]].copy()
    out["sign"] = np.where(out["rel_dev"] >= 0, "+", "−")
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------- 2. паспорт типа в исходных единицах
def _split_raw_names(raw_names: list[str], level_names: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Делит исходные признаки на доли (0..1) и уровни (₽): по явному списку или по имени (level/total/итог)."""
    if level_names is None:
        level_names = [n for n in raw_names if any(s in n.lower() for s in ("level", "total", "итог", "всего"))]
    shares = [n for n in raw_names if n not in level_names]
    return shares, list(level_names)


def readable_profile(raw: np.ndarray, labels, raw_names: list[str], level_names: list[str] | None = None,
                     cfg: dict | None = None) -> pd.DataFrame:
    """Медианы по типам и по всей выборке в исходных единицах: доли → %, уровень → ₽ и % к медиане страны.

    Строки — типы и «Все МО»; столбцы: n, «<доля>, %», «<уровень>, ₽», «<уровень>, % к медиане».
    """
    rcfg = (cfg or {}).get("readable", {})
    suffix = rcfg.get("share_suffix", ", %")
    unit = rcfg.get("level_unit", "₽")
    raw = np.asarray(raw, float)
    enc, types = _labels(labels)
    shares, levels = _split_raw_names(raw_names, level_names)
    col = {n: i for i, n in enumerate(raw_names)}
    med_all = np.nanmedian(raw, axis=0)
    rows = []
    for j, t in [(j, t) for j, t in enumerate(types)] + [(None, ALL_LABEL)]:
        sub = raw if j is None else raw[enc == j]
        med = np.nanmedian(sub, axis=0)
        r: dict[str, Any] = {"type": t, "n": int(sub.shape[0])}
        for n in shares:
            r[f"{n}{suffix}"] = 100 * med[col[n]]
        for n in levels:
            r[f"{n}, {unit}"] = med[col[n]]
            r[f"{n}, % к медиане"] = 100 * med[col[n]] / med_all[col[n]] if med_all[col[n]] else np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("type")


# ----------------------------------------------------------------------------- 3. внешняя проверка
def kruskal_eta(values, labels) -> dict:
    """Краскел–Уоллис по типам и размер эффекта η²_H = (H − k + 1)/(n − k) (Tomczak & Tomczak 2014), обрезан снизу нулём.
    Потолок η²_H при идеальном разделении k равных групп ≈ 1 − 1/k² (ранговая статистика), при k=3 это 0.89.
    NaN в values отбрасываются; при < 2 непустых групп возвращаются NaN."""
    v = np.asarray(pd.Series(values).to_numpy(), float)
    lab = np.asarray(pd.Series(labels).to_numpy())
    ok = ~np.isnan(v)
    v, lab = v[ok], lab[ok]
    groups = [v[lab == t] for t in pd.unique(lab)]
    groups = [g for g in groups if len(g) > 0]
    n, k = len(v), len(groups)
    if k < 2 or n <= k or np.all(v == v[0]):
        return {"H": np.nan, "p": np.nan, "eta2_H": np.nan, "n": n, "k": k}
    H, p = stats.kruskal(*groups)
    eta2 = max(0.0, (H - k + 1) / (n - k))
    return {"H": float(H), "p": float(p), "eta2_H": float(min(eta2, 1.0)), "n": n, "k": k}


def _collapse_rare(s: pd.Series, min_size: int) -> pd.Series:
    """Редкие категории → «прочее», иначе таблица сопряжённости вырождается."""
    cnt = s.value_counts()
    rare = set(cnt[cnt < min_size].index)
    return s.where(~s.isin(rare), "прочее") if rare else s


def cramers_v(cat, labels, min_size: int = 1) -> dict:
    """V Крамера (с поправкой Бергсмы не пользуемся — scipy даёт классический), NMI и таблица сопряжённости."""
    s = pd.Series(np.asarray(cat)).astype("string")
    lab = pd.Series(np.asarray(labels)).astype("string")
    ok = s.notna() & lab.notna()
    s, lab = s[ok], lab[ok]
    s = _collapse_rare(s, min_size)
    ct = pd.crosstab(s, lab)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"V": np.nan, "p": np.nan, "nmi": np.nan, "chi2": np.nan, "table": ct, "n": int(ok.sum())}
    chi2, p, _, _ = stats.chi2_contingency(ct.to_numpy())
    V = association(ct.to_numpy(), method="cramer")
    nmi = normalized_mutual_info_score(s.to_numpy(), lab.to_numpy())
    return {"V": float(V), "p": float(p), "nmi": float(nmi), "chi2": float(chi2), "table": ct, "n": int(ok.sum())}


def external_validation(external_df: pd.DataFrame, labels_by_territory: pd.Series, meta: pd.DataFrame | None = None,
                        cfg: dict | None = None) -> pd.DataFrame:
    """Внешняя проверка типологии переменными, которые НЕ входили в кластеризацию.

    external_df — таблица по territory_id (индекс или колонка): числовые колонки → Краскел–Уоллис (H, p, η²_H)
    и медианы по типам; строковые/категориальные (и meta: municipal_district_type, region_name,
    municipal_district_status) → V Крамера, NMI, χ²; таблицы сопряжённости — в attrs["tables"][variable].
    labels_by_territory — Series: индекс territory_id, значения — тип. Результат отсортирован по силе связи
    (strength = η²_H для числовых, V для категориальных).
    """
    ecfg = (cfg or {}).get("external", {})
    meta_cat = ecfg.get("meta_categorical", ["municipal_district_type", "region_name", "municipal_district_status"])
    min_size = int(ecfg.get("min_category_size", 5))
    lab = pd.Series(labels_by_territory).dropna()
    lab.index = lab.index.astype(int)

    frames = []
    ext = external_df.copy()
    if "territory_id" in ext.columns:
        ext = ext.set_index("territory_id")
    ext.index = ext.index.astype(int)
    frames.append(("external", ext))
    if meta is not None:
        m = meta.copy()
        if "territory_id" in m.columns:
            m = m.set_index("territory_id")
        m.index = m.index.astype(int)
        cols = [c for c in meta_cat if c in m.columns]
        frames.append(("meta", m[cols]))

    types = sorted(pd.unique(lab).tolist())
    rows, tables = [], {}
    for source, df in frames:
        df = df.reindex(lab.index)
        for c in df.columns:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                r = kruskal_eta(s, lab)
                row = {"variable": c, "source": source, "kind": "numeric", "stat": "eta2_H", "strength": r["eta2_H"],
                       "p": r["p"], "H": r["H"], "V": np.nan, "nmi": np.nan, "n": r["n"]}
                for t in types:
                    row[f"median_{t}"] = float(np.nanmedian(s[lab == t].to_numpy(float))) if (lab == t).any() else np.nan
                row["median_all"] = float(np.nanmedian(s.to_numpy(float)))
            else:
                r = cramers_v(s, lab, min_size)
                row = {"variable": c, "source": source, "kind": "categorical", "stat": "cramers_v", "strength": r["V"],
                       "p": r["p"], "H": np.nan, "V": r["V"], "nmi": r["nmi"], "n": r["n"]}
                tables[c] = r["table"]
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("strength", ascending=False, na_position="last").reset_index(drop=True)
    out.attrs["tables"] = tables
    out.attrs["types"] = types
    return out


# ----------------------------------------------------------------------------- 4. дерево-суррогат
def readable_feature_names(raw_names: list[str], level_names: list[str] | None = None, cfg: dict | None = None) -> list[str]:
    """Русские подписи признаков в исходных единицах: «Доля «Продовольствие», %», «Итог на жителя, ₽»."""
    rcfg = (cfg or {}).get("readable", {})
    unit = rcfg.get("level_unit", "₽")
    level_label = rcfg.get("level_name", "Итог на жителя")
    shares, levels = _split_raw_names(raw_names, level_names)
    out = []
    for n in raw_names:
        if n in levels:
            out.append(f"{level_label}, {unit}" if len(levels) == 1 else f"{n}, {unit}")
        else:
            out.append(f"Доля «{n.replace('share_', '')}», %")
    return out


def rule_tree(raw: np.ndarray, labels, raw_names: list[str], max_depth: int = 3, min_leaf: int = 30,
              cv_folds: int = 5, seed: int = 42, level_names: list[str] | None = None,
              type_names: dict | None = None, cfg: dict | None = None) -> dict:
    """Дерево-суррогат (DecisionTreeClassifier) на исходных единицах: правила в процентах и рублях.

    Возвращает dict(text — правила export_text с русскими именами признаков, accuracy — на обучении,
    cv_accuracy — по стратифицированной k-fold CV, tree — обученная модель, feature_names, class_names,
    leaf_purity — доля главного класса по листьям).
    """
    raw = np.asarray(raw, float)
    enc, types = _labels(labels)
    shares, levels = _split_raw_names(raw_names, level_names)
    Z = raw.copy()
    for i, n in enumerate(raw_names):
        if n in shares:
            Z[:, i] = 100 * Z[:, i]  # доли → проценты, чтобы пороги читались
    fnames = readable_feature_names(raw_names, levels, cfg)
    cnames = [str((type_names or {}).get(t, {}).get("short", t)) if isinstance((type_names or {}).get(t), dict)
              else str((type_names or {}).get(t, t)) for t in types]
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=seed).fit(Z, enc)
    acc = float(tree.score(Z, enc))
    counts = np.bincount(enc, minlength=len(types))
    folds = int(min(cv_folds, counts.min())) if counts.min() >= 2 else 0
    if folds >= 2:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=seed)
        cv_acc = float(cross_val_score(clf, Z, enc, cv=cv).mean())
    else:
        cv_acc = float("nan")
    text = export_text(tree, feature_names=fnames, class_names=cnames, decimals=1, show_weights=False)
    leaf = tree.apply(Z)
    purity = {int(l): float(np.bincount(enc[leaf == l]).max() / (leaf == l).sum()) for l in np.unique(leaf)}
    return {"text": text, "accuracy": acc, "cv_accuracy": cv_acc, "tree": tree, "feature_names": fnames,
            "class_names": cnames, "leaf_purity": purity, "types": types}


# ----------------------------------------------------------------------------- 5. типичные и пограничные МО
def _two_nearest(X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Расстояния до ближайшего и второго ближайшего центра и их индексы."""
    D = np.sqrt(((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    order = np.argsort(D, axis=1)
    i1, i2 = order[:, 0], order[:, 1]
    return D[np.arange(len(X)), i1], D[np.arange(len(X)), i2], i1, i2


def membership_margin(X: np.ndarray, labels, centers: np.ndarray | None = None) -> np.ndarray:
    """Нормированный зазор принадлежности m_i = (d₂ − d₁)/(d₁ + d₂) ∈ [0, 1]: 0 — на границе двух центров,
    1 — в самом центре. d₁ — расстояние до центра своего типа, d₂ — до ближайшего чужого.
    Для слоя неопределённости на карте (низкий зазор = ненадёжная принадлежность)."""
    X = np.asarray(X, float)
    enc, types = _labels(labels)
    C = _centers(X, enc, len(types)) if centers is None else np.asarray(centers, float)
    D = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))
    d_own = D[np.arange(len(X)), enc]
    D_other = D.copy()
    D_other[np.arange(len(X)), enc] = np.inf
    d_other = D_other.min(axis=1)
    denom = d_own + d_other
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(denom > 0, (d_other - d_own) / denom, 1.0)
    return np.clip(m, -1.0, 1.0)


def typical_and_border(X: np.ndarray, labels, meta: pd.DataFrame, n: int = 3,
                       centers: np.ndarray | None = None) -> pd.DataFrame:
    """Для каждого типа: n МО ближе всех к центру (role='typical') и n МО с наименьшим зазором между
    расстоянием до своего и до ближайшего чужого центра (role='border'). meta — строки в порядке X
    (territory_id, name, region_name, ... — берутся те, что есть)."""
    X = np.asarray(X, float)
    enc, types = _labels(labels)
    C = _centers(X, enc, len(types)) if centers is None else np.asarray(centers, float)
    D = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))
    d_own = D[np.arange(len(X)), enc]
    D_other = D.copy()
    D_other[np.arange(len(X)), enc] = np.inf
    second = D_other.argmin(axis=1)
    d_other = D_other.min(axis=1)
    margin = membership_margin(X, labels, C)
    meta_cols = [c for c in ("territory_id", "name", "region_name", "municipal_district_type") if c in meta.columns]
    m = meta.reset_index(drop=True)
    rows = []
    for j, t in enumerate(types):
        idx = np.where(enc == j)[0]
        typ = idx[np.argsort(d_own[idx])][:n]
        bor = idx[np.argsort(margin[idx])][:n]
        for role, sel in (("typical", typ), ("border", bor)):
            for r, i in enumerate(sel, 1):
                row = {"type": t, "role": role, "rank": r, "row": int(i)}
                row.update({c: m.at[i, c] for c in meta_cols})
                row.update({"dist_own": float(d_own[i]), "dist_second": float(d_other[i]),
                            "margin": float(margin[i]), "second_type": types[int(second[i])]})
                rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 6. имена типов
def _resolve_threshold(v, thresholds: dict) -> float:
    return float(thresholds[v]) if isinstance(v, str) else float(v)


def _rule_violations(rule: dict, means: pd.Series, thresholds: dict, labels_ru: dict) -> list[str]:
    """Список нарушений условий правила для вектора средних типа (пустой — правило выполнено)."""
    out = []
    for feat, cond in rule.get("when", {}).items():
        if feat not in means.index:
            out.append(f"нет признака «{feat}»")
            continue
        val = float(means[feat])
        lab = labels_ru.get(feat, feat)
        if "min" in cond and val < _resolve_threshold(cond["min"], thresholds):
            out.append(f"{lab}: {val:+.2f}σ < порога {_resolve_threshold(cond['min'], thresholds):+.2f}σ")
        if "max" in cond and val > _resolve_threshold(cond["max"], thresholds):
            out.append(f"{lab}: {val:+.2f}σ > порога {_resolve_threshold(cond['max'], thresholds):+.2f}σ")
    return out


def _describe_from_readable(k, readable: pd.DataFrame | None) -> str:
    """Одна фраза с медианами в исходных единицах для описания типа."""
    if readable is None or k not in readable.index or ALL_LABEL not in readable.index:
        return ""
    r, a = readable.loc[k], readable.loc[ALL_LABEL]
    parts = []
    for c in readable.columns:
        if c.endswith(", % к медиане"):
            parts.insert(0, f"итог трат {r[c]:.0f} % к медиане страны")
        elif c.endswith(", %") and pd.notna(r[c]) and pd.notna(a[c]) and a[c] > 0:
            diff = r[c] - a[c]
            if abs(diff) >= 1.0:
                word = "выше" if diff > 0 else "ниже"
                parts.append(f"доля «{c[:-3]}» {r[c]:.1f} % ({word} общей на {abs(diff):.1f} п.п.)")
    return ("Медианы: " + "; ".join(parts) + ".") if parts else ""


def name_types(profile_std: pd.DataFrame, readable: pd.DataFrame | None, rules: dict | None, cfg: dict | None = None) -> dict:
    """Черновые имена типов по правилам configs/interpret.yaml → {k: {name, short, description, rule, top_features}}.

    Условия правил — на средние стандартизованных признаков (profile_std из mirkin_profile); первое подходящее
    правило выигрывает; если правило уже занято другим типом, добавляется суффикс (II, III). Если ни одно правило
    не подошло — имя по топ-признакам Миркина. Имена — черновик; финальные подтверждает человек (check_names).
    `rules` — результат rule_tree (используется для описания), может быть None.
    """
    cfg = cfg or load_config()
    ncfg = cfg["naming"]
    thresholds = ncfg.get("thresholds", {})
    labels_ru = ncfg.get("feature_labels", {})
    wide = profile_wide(profile_std, "mean")
    out, used = {}, {}
    roman = ["", " II", " III", " IV", " V", " VI", " VII", " VIII"]
    for k in wide.index:
        means = wide.loc[k]
        chosen = None
        for rule in ncfg.get("rules", []):
            if not _rule_violations(rule, means, thresholds, labels_ru):
                chosen = rule
                break
        tf = top_features(profile_std, k, cfg.get("mirkin", {}).get("top_n", 3))
        feats = ", ".join(f"{labels_ru.get(f, f)} {s}" for f, s in zip(tf["feature"], tf["sign"]))
        if chosen is None:
            name = ncfg.get("fallback_name", "Тип {k}: {features}").format(k=k, features=feats)
            short, rule_name, base_desc = f"Тип {k}", None, f"Отличительные признаки: {feats}."
        else:
            cnt = used.get(chosen["name"], 0)
            used[chosen["name"]] = cnt + 1
            name = chosen["name"] + roman[min(cnt, len(roman) - 1)]
            short = chosen.get("short", chosen["name"]) + roman[min(cnt, len(roman) - 1)]
            rule_name, base_desc = chosen["name"], chosen.get("description", "")
        desc = " ".join(s for s in (base_desc, _describe_from_readable(k, readable)) if s)
        out[k] = {"name": name, "short": short, "description": desc, "rule": rule_name,
                  "top_features": tf["feature"].tolist()}
    return out


def check_names(profile: pd.DataFrame, names_dict: dict, cfg: dict | None = None) -> list[str]:
    """Проверка кодом: заявленное имя типа согласуется с его профилем. Для каждого типа ищется правило с таким
    именем (поле rule, иначе name/short без суффикса II/III); его условия проверяются по средним профиля.
    Возвращает список нарушений (пустой — ок). Имена, не описанные правилами, не проверяются."""
    cfg = cfg or load_config()
    ncfg = cfg["naming"]
    thresholds = ncfg.get("thresholds", {})
    labels_ru = ncfg.get("feature_labels", {})
    by_name = {r["name"]: r for r in ncfg.get("rules", [])}
    by_short = {r.get("short", r["name"]): r for r in ncfg.get("rules", [])}
    wide = profile_wide(profile, "mean")
    problems = []
    for k, info in names_dict.items():
        if k not in wide.index:
            problems.append(f"тип {k}: нет в профиле")
            continue
        info = info if isinstance(info, dict) else {"name": str(info)}
        cands = [info.get("rule"), info.get("name"), info.get("short")]
        rule = None
        for c in cands:
            if not c:
                continue
            # снимаем суффикс-дубль из name_types (II…VIII); раньше снимались только II/III/IV,
            # и имена «… V», «… VI» молча не проверялись
            base = re.sub(r"\s+(?:II|III|IV|V|VI|VII|VIII)$", "", str(c).strip())
            rule = by_name.get(base) or by_short.get(base)
            if rule:
                break
        if rule is None:
            continue
        viol = _rule_violations(rule, wide.loc[k], thresholds, labels_ru)
        problems += [f"тип {k} «{info.get('name')}»: {v}" for v in viol]
    return problems


# ----------------------------------------------------------------------------- 7. SHAP-важности
def shap_importance(X: np.ndarray, labels, names: list[str], seed: int = 42) -> pd.DataFrame | None:
    """LightGBM multiclass + SHAP: средний |SHAP| признак × тип (строки — типы, столбцы — признаки).
    Если lightgbm/shap недоступны — предупреждение и None."""
    try:
        import lightgbm as lgb
        import shap
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"shap_importance пропущен: {e}")
        return None
    X = np.asarray(X, float)
    enc, types = _labels(labels)
    k = len(types)
    params = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=5,
                  random_state=seed, verbose=-1)
    model = lgb.LGBMClassifier(objective="multiclass" if k > 2 else "binary", **params).fit(X, enc)
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):
        arr = np.stack(sv, axis=-1)  # (N, F, K)
    else:
        arr = np.asarray(sv)
    if arr.ndim == 2:  # бинарный случай: вклад для класса 1; для класса 0 — с обратным знаком
        arr = np.stack([-arr, arr], axis=-1)
    imp = np.abs(arr).mean(axis=0).T  # (K, F)
    return pd.DataFrame(imp, index=pd.Index(types, name="type"), columns=names)


# ----------------------------------------------------------------------------- 8. сводка для лендинга
def type_summary(labels, meta: pd.DataFrame, profile: pd.DataFrame, readable: pd.DataFrame,
                 typical_border: pd.DataFrame, names_dict: dict, external_val: pd.DataFrame | None = None,
                 rules: dict | None = None, shap_df: pd.DataFrame | None = None, top_n: int = 3) -> dict:
    """Собирает всё в JSON-сериализуемую структуру: по типу — имя, описание, n, доля, профиль, медианы,
    топ-признаки, типичные/пограничные МО, медианы внешних переменных; плюс общие показатели (B/T, дерево)."""
    enc, types = _labels(labels)
    N = len(enc)
    out: dict[str, Any] = {"n_total": N, "k": len(types), "B_over_T": profile.attrs.get("B_over_T"),
                           "cluster_contrib": profile.attrs.get("cluster_contrib", {}), "types": []}
    if rules is not None:
        out["rule_tree"] = {"text": rules["text"], "accuracy": rules["accuracy"], "cv_accuracy": rules["cv_accuracy"]}
    tb_cols = [c for c in ("territory_id", "name", "region_name", "margin", "dist_own", "second_type") if c in typical_border.columns]
    for j, t in enumerate(types):
        info = names_dict.get(t, {"name": f"Тип {t}", "short": f"Тип {t}", "description": ""})
        prof = profile[profile["type"] == t].set_index("feature")[["mean", "rel_dev", "contrib_pct_T", "contrib_pct_B"]]
        tb = typical_border[typical_border["type"] == t]
        entry = {"type": t, "name": info.get("name"), "short": info.get("short"), "description": info.get("description"),
                 "n": int((enc == j).sum()), "share": float((enc == j).mean()),
                 "profile": prof.to_dict(orient="index"),
                 "medians": readable.loc[t].to_dict() if t in readable.index else {},
                 "top_features": top_features(profile, t, top_n).to_dict(orient="records"),
                 "typical": tb[tb["role"] == "typical"][tb_cols].to_dict(orient="records"),
                 "border": tb[tb["role"] == "border"][tb_cols].to_dict(orient="records")}
        if external_val is not None and len(external_val):
            num = external_val[external_val["kind"] == "numeric"]
            col = f"median_{t}"
            if col in num.columns:
                entry["external_medians"] = dict(zip(num["variable"], num[col]))
        if shap_df is not None and t in shap_df.index:
            entry["shap"] = shap_df.loc[t].to_dict()
        out["types"].append(entry)
    if readable is not None and ALL_LABEL in readable.index:
        out["medians_all"] = readable.loc[ALL_LABEL].to_dict()
    if external_val is not None and len(external_val):
        out["external"] = external_val.drop(columns=[c for c in external_val.columns if c.startswith("median_")]).to_dict(orient="records")
    return _to_py(out)


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    """Конвейер на реальных артефактах: признаки из outputs/features, метки — из файла (--labels)."""
    import argparse

    from pulsar import data

    ap = argparse.ArgumentParser(description="Интерпретация типов МО")
    ap.add_argument("--labels", default=str(ROOT / "outputs" / "dynamics" / "labels.npy"), help=".npy с метками по ids панели")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    feat = ROOT / "outputs" / "features"
    lab_path = pathlib.Path(args.labels)
    if not lab_path.exists() or not (feat / "X.npy").exists():
        print(f"Нет {lab_path} или {feat / 'X.npy'} — сначала `make dynamics` (или передайте --labels).")
        return
    cfg = load_config()
    X = np.load(feat / "X.npy")
    X = X[-1] if X.ndim == 3 else X  # последний месяц, если передан куб T×N×F
    ids = np.load(feat / "ids.npy")
    names = json.loads((feat / "names.json").read_text(encoding="utf-8"))
    labels = np.load(lab_path, allow_pickle=True)
    mdict = data.read_municipal_dict()
    meta = mdict.set_index("territory_id").reindex(ids).reset_index()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prof = mirkin_profile(X, labels, names)
    prof.to_csv(out / "profile.csv", index=False)
    tb = typical_and_border(X, labels, meta, cfg["typical"]["n"])
    tb.to_csv(out / "typical_border.csv", index=False)
    names_dict = name_types(prof, None, None, cfg)
    readable = pd.DataFrame(index=pd.Index(list(names_dict) + [ALL_LABEL], name="type"))
    summary = type_summary(labels, meta, prof, readable, tb, names_dict)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"B/T = {prof.attrs['B_over_T']:.3f}; типы: " + "; ".join(f"{k}: {v['name']}" for k, v in names_dict.items()))
    for p in check_names(prof, names_dict, cfg):
        print("ВНИМАНИЕ:", p)


if __name__ == "__main__":
    main()
