"""Признаки МО по месяцам и матрица расстояний.

Признаки месяца t для МО i (F = 7):
  CLR долей шести категорий (5 категорий + «Прочее» = итог − сумма пяти):  clr_c = ln(s_c) − mean_c ln(s_c)
  лог-уровень: ln(итог_it / медиана_t по стране)
Затем причинное сглаживание окном 3 (t−2..t) и z-стандартизация внутри месяца.

Запуск: `python -m pulsar.features` → outputs/features/{X.npy, X_all.npy, ids.npy, ids_all.npy, months.json,
names.json, levels.npy, growth.npy, D.npy, panel.parquet}.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from pulsar import data

ROOT = data.ROOT
OUT = ROOT / "outputs" / "features"


def wide_table(cons: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Таблица (territory_id, date) × категории, включая итог и «Прочее»."""
    w = cons.pivot_table(index=["territory_id", "date"], columns="category", values="value", aggfunc="first")
    cats = cfg["categories"]
    w[cfg["other_name"]] = w[cfg["total"]] - w[cats].sum(axis=1)
    return w


def select_panel(w: pd.DataFrame, cfg: dict, mdict: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (ids_panel, ids_all): МО с полной историей и все МО."""
    months_per_id = w.groupby(level=0).size()
    ids_all = np.array(sorted(months_per_id.index))
    full = months_per_id[months_per_id >= cfg["panel"]["min_months"]].index
    ids = np.array(sorted(full))
    if cfg["panel"].get("drop_intracity"):
        intra = set(mdict.loc[mdict.municipal_district_type.str.contains("внутригородск", na=False), "territory_id"])
        ids = np.array([i for i in ids if i not in intra])
    return ids, ids_all


def raw_features(w: pd.DataFrame, cfg: dict, months: list[str], ids: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Сырые признаки (T, N, F) до сглаживания и стандартизации; NaN там, где МО нет в месяце."""
    comps = cfg["categories"] + [cfg["other_name"]]
    total = cfg["total"]
    T, N, F = len(months), len(ids), len(comps) + 1
    X = np.full((T, N, F), np.nan, dtype=np.float64)
    levels = np.full((T, N), np.nan)
    idx = {tid: j for j, tid in enumerate(ids)}
    eps = cfg["composition"]["pseudocount"]
    for t, m in enumerate(months):
        try:
            sub = w.xs(m, level="date")
        except KeyError:
            continue
        sub = sub[sub.index.isin(ids)]
        rows = np.array([idx[i] for i in sub.index])
        shares = sub[comps].to_numpy(float) / sub[[total]].to_numpy(float)
        shares = np.clip(shares, eps, None)
        shares = shares / shares.sum(axis=1, keepdims=True)
        logs = np.log(shares)
        clr = logs - logs.mean(axis=1, keepdims=True)
        tot = sub[total].to_numpy(float)
        med = np.nanmedian(tot)
        X[t, rows, : len(comps)] = clr
        X[t, rows, len(comps)] = np.log(tot / med) * cfg["level"]["weight"]
        levels[t, rows] = tot
    names = [f"clr_{c}" for c in comps] + ["log_level_rel"]
    return X, names, levels


def causal_smooth(X: np.ndarray, window: int) -> np.ndarray:
    """Среднее по доступным значениям за окно t−window+1..t (не использует будущее)."""
    if window <= 1:
        return X.copy()
    out = np.full_like(X, np.nan)
    for t in range(X.shape[0]):
        lo = max(0, t - window + 1)
        out[t] = np.nanmean(X[lo : t + 1], axis=0)
    return out


def standardize_within_month(X: np.ndarray) -> np.ndarray:
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


def growth_matrix(levels: np.ndarray) -> np.ndarray:
    """Лог-приросты итоговых трат (T−1, N); для со-движения из них вычитается медиана страны в месяце."""
    g = np.diff(np.log(levels), axis=0)
    return g - np.nanmedian(g, axis=1, keepdims=True)


def centroids_from_gpkg(cfg_data: dict | None = None) -> pd.DataFrame:
    """Центроиды полигонов МО (lat, lon) из gpkg СберИндекса; кэшируются в outputs/features/centroids.parquet."""
    cache = OUT / "centroids.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import geopandas as gpd
    cfg_data = cfg_data or data.load_config("data")
    g = gpd.read_file(ROOT / cfg_data["municipal_dict"]["gpkg"], columns=["territory_id", "year_from", "year_to"])
    g = g.sort_values("year_to").drop_duplicates("territory_id", keep="last")
    pts = g.to_crs(3576).geometry.representative_point().to_crs(4326)  # полярная равноплощадная проекция → корректно и для Чукотки
    out = pd.DataFrame({"territory_id": pd.to_numeric(g.territory_id).astype(int).to_numpy(), "lat": pts.y.to_numpy(), "lon": pts.x.to_numpy()})
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def centers(ids: np.ndarray, mdict: pd.DataFrame, cfg: dict | None = None) -> np.ndarray:
    """Координаты (lat, lon) МО: центр из справочника, иначе центроид полигона."""
    m = mdict.set_index("territory_id").reindex(ids)
    lat = m["municipal_district_center_lat"].to_numpy(float)
    lon = m["municipal_district_center_lon"].to_numpy(float)
    miss = np.isnan(lat) | np.isnan(lon)
    if miss.any():
        c = centroids_from_gpkg().set_index("territory_id").reindex(ids)
        lat = np.where(miss, c["lat"].to_numpy(float), lat)
        lon = np.where(miss, c["lon"].to_numpy(float), lon)
    return np.column_stack([lat, lon])


def build_distance_matrix(ids: np.ndarray, cfg: dict, mdict: pd.DataFrame) -> np.ndarray:
    """Симметричная матрица расстояний (км) между центрами МО панели."""
    dcfg = cfg["distance"]
    conn = data.read_connection(kind=None)
    idx = {tid: j for j, tid in enumerate(ids)}
    N = len(ids)
    D = np.full((N, N), np.nan)
    for kind in (dcfg["kind"], dcfg["fallback"]):
        sub = conn[(conn["type"] == kind) & conn.territory_id_x.isin(idx) & conn.territory_id_y.isin(idx)]
        a = sub.territory_id_x.map(idx).to_numpy(); b = sub.territory_id_y.map(idx).to_numpy()
        d = sub.distance.to_numpy(float)
        mask = np.isnan(D[a, b])
        D[a[mask], b[mask]] = d[mask]
        D[b[mask], a[mask]] = d[mask]
    # остаток — большой круг × поправка; координаты центра, при их отсутствии — центроид полигона
    cen = centers(ids, mdict, cfg)
    lat, lon = np.radians(cen[:, 0]), np.radians(cen[:, 1])
    dlat = lat[:, None] - lat[None, :]; dlon = lon[:, None] - lon[None, :]
    h = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    gc = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1))) * dcfg["missing_km_factor"]
    miss = np.isnan(D)
    D[miss] = gc[miss]
    np.fill_diagonal(D, 0.0)
    off = ~np.eye(N, dtype=bool)
    D[off & (D <= 0)] = dcfg["zero_distance_km"]
    return D


def build(cfg: dict | None = None, write: bool = True) -> dict:
    cfg = cfg or data.load_config("features")
    cons = data.read_consumption()
    mdict = data.read_municipal_dict()
    w = wide_table(cons, cfg)
    months = sorted(cons.date.unique())
    ids, ids_all = select_panel(w, cfg, mdict)

    X_all_raw, names, levels_all = raw_features(w, cfg, months, ids_all)
    X_all = standardize_within_month(causal_smooth(X_all_raw, cfg["smoothing"]["window"]))
    sel = np.isin(ids_all, ids)
    X = X_all[:, sel, :]
    levels = levels_all[:, sel]
    growth = growth_matrix(levels)
    D = build_distance_matrix(ids, cfg, mdict)

    panel = (pd.DataFrame({"territory_id": ids_all, "in_panel": sel})
             .merge(mdict[["territory_id", "municipal_district_name_short", "municipal_district_name", "municipal_district_type",
                           "municipal_district_status", "region_code", "region_name", "oktmo8",
                           "municipal_district_center_lat", "municipal_district_center_lon"]], on="territory_id", how="left"))
    cen_all = centers(ids_all, mdict, cfg)
    panel["lat"], panel["lon"] = cen_all[:, 0], cen_all[:, 1]
    panel["months_present"] = pd.Series(np.isfinite(levels_all).sum(axis=0), index=range(len(ids_all))).to_numpy()

    out = {"X": X.astype(np.float32), "X_all": X_all.astype(np.float32), "ids": ids, "ids_all": ids_all, "months": months,
           "names": names, "levels": levels, "growth": growth, "D": D.astype(np.float32), "panel": panel}
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        for key in ("X", "X_all", "ids", "ids_all", "levels", "growth", "D"):
            np.save(OUT / f"{key}.npy", out[key])
        (OUT / "months.json").write_text(json.dumps(months, ensure_ascii=False))
        (OUT / "names.json").write_text(json.dumps(names, ensure_ascii=False))
        panel.to_parquet(OUT / "panel.parquet", index=False)
    return out


def load() -> dict:
    out = {key: np.load(OUT / f"{key}.npy", allow_pickle=False) for key in ("X", "X_all", "ids", "ids_all", "levels", "growth", "D")}
    out["months"] = json.loads((OUT / "months.json").read_text())
    out["names"] = json.loads((OUT / "names.json").read_text())
    out["panel"] = pd.read_parquet(OUT / "panel.parquet")
    return out


if __name__ == "__main__":
    res = build()
    X = res["X"]
    print(f"панель: {len(res['ids'])} МО × {len(res['months'])} мес. × {X.shape[2]} признаков; NaN в панели: {int(np.isnan(X).sum())}")
    print(f"все МО: {len(res['ids_all'])}; D: {res['D'].shape}, медиана {np.median(res['D'][np.triu_indices(len(res['ids']), 1)]):.0f} км")
