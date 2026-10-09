"""Тесты признаков: CLR, причинное сглаживание, стандартизация, матрица расстояний."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from pulsar import features  # noqa: E402

CFG = {
    "panel": {"min_months": 3, "drop_intracity": False},
    "categories": ["A", "B"], "total": "Все категории", "other_name": "Прочее",
    "composition": {"transform": "clr", "pseudocount": 1e-4},
    "level": {"transform": "log_rel_median", "weight": 1.0},
    "smoothing": {"window": 3}, "standardize": "within_month",
    "distance": {"kind": "highway", "fallback": "railway", "zero_distance_km": 5.0, "missing_km_factor": 1.3},
}


def toy_consumption() -> pd.DataFrame:
    rows = []
    for tid, (a, b, tot) in {1: (50, 20, 100), 2: (60, 10, 200), 3: (30, 30, 50)}.items():
        for m, mult in zip(["2023-01", "2023-02", "2023-03"], [1.0, 1.1, 1.3]):
            rows += [(m, tid, "A", a * mult), (m, tid, "B", b * mult), (m, tid, "Все категории", tot * mult)]
    rows += [("2023-01", 4, "A", 10), ("2023-01", 4, "B", 10), ("2023-01", 4, "Все категории", 40)]  # неполная история
    return pd.DataFrame(rows, columns=["date", "territory_id", "category", "value"])


def test_clr_rows_sum_to_zero_and_level_relative():
    cons = toy_consumption()
    w = features.wide_table(cons, CFG)
    months = ["2023-01", "2023-02", "2023-03"]
    ids = np.array([1, 2, 3])
    X, names, levels = features.raw_features(w, CFG, months, ids)
    assert names == ["clr_A", "clr_B", "clr_Прочее", "log_level_rel"]
    assert np.allclose(X[:, :, :3].sum(axis=2), 0.0, atol=1e-9)
    # уровень относительно медианы: у МО 1 (100 при медиане 100) → 0
    assert X[0, 0, 3] == pytest.approx(0.0, abs=1e-9)
    assert X[0, 1, 3] == pytest.approx(np.log(2.0))
    # общий мультипликативный рост не меняет ни CLR, ни относительный уровень
    assert np.allclose(X[0], X[2], atol=1e-9)


def test_panel_selection_requires_full_history():
    cons = toy_consumption()
    w = features.wide_table(cons, CFG)
    md = pd.DataFrame({"territory_id": [1, 2, 3, 4], "municipal_district_type": ["район"] * 4})
    ids, ids_all = features.select_panel(w, CFG, md)
    assert list(ids) == [1, 2, 3] and list(ids_all) == [1, 2, 3, 4]


def test_causal_smoothing_uses_only_past():
    X = np.arange(5, dtype=float).reshape(5, 1, 1)
    S = features.causal_smooth(X, 3)
    assert S[0, 0, 0] == 0 and S[1, 0, 0] == 0.5 and S[4, 0, 0] == 3.0
    X2 = X.copy(); X2[4] = 100  # изменение будущего не влияет на прошлое
    assert np.allclose(features.causal_smooth(X2, 3)[:4], S[:4])


def test_standardize_within_month():
    rng = np.random.default_rng(0)
    X = rng.normal(5, 3, size=(4, 50, 3))
    Z = features.standardize_within_month(X)
    assert np.allclose(Z.mean(axis=1), 0, atol=1e-9) and np.allclose(Z.std(axis=1), 1, atol=1e-9)


def test_growth_matrix_removes_common_trend():
    levels = np.array([[100, 200, 300], [110, 220, 330], [121, 242, 363]], dtype=float)
    g = features.growth_matrix(levels)
    assert g.shape == (2, 3) and np.allclose(g, 0.0, atol=1e-12)


def test_distance_matrix_fallbacks(monkeypatch):
    conn = pd.DataFrame({
        "territory_id_x": [2, 3, 3], "territory_id_y": [1, 1, 2],
        "distance": [100.0, 0.0, 50.0], "type": ["highway", "highway", "railway"],
    })
    monkeypatch.setattr(features.data, "read_connection", lambda kind=None: conn)
    md = pd.DataFrame({"territory_id": [1, 2, 3], "municipal_district_center_lat": [55.0, 55.0, 56.0],
                       "municipal_district_center_lon": [37.0, 38.0, 37.0]})
    D = features.build_distance_matrix(np.array([1, 2, 3]), CFG, md)
    assert np.allclose(D, D.T) and np.all(np.diag(D) == 0)
    assert D[0, 1] == 100.0            # автодорога
    assert D[0, 2] == 5.0              # нулевое расстояние → 5 км
    assert D[1, 2] == 50.0             # ж/д как запасной вариант
