"""Тесты модуля оценки: нулевые модели, бутстрэп, рейтинги."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from pulsar import evaluate  # noqa: E402


def blobs_and_graph(seed=0, k=4, n_per=40):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(4 * c, 1.0, size=(n_per, 3)) for c in range(k)])
    lab = np.repeat(np.arange(k), n_per)
    inside = lab[:, None] == lab[None, :]
    A = (rng.random((len(lab), len(lab))) < np.where(inside, 0.3, 0.01)).astype(float)
    A = np.triu(A, 1); A = A + A.T
    return X, A, lab


def test_metric_panel_keys_and_mq_is_modularity():
    X, A, lab = blobs_and_graph()
    m = evaluate.metric_panel(X, A, lab)
    for key in evaluate.PANEL:
        assert key in m
    assert m["MQ"] == pytest.approx(__import__("pulsar.icvi", fromlist=["modularity"]).modularity(A, lab))
    assert m["CH_N"] == pytest.approx(m["CH"] / len(lab))


def test_permutation_null_detects_structure():
    X, A, lab = blobs_and_graph()
    r = evaluate.permutation_null(X, A, lab, n_perm=30, seed=1).set_index("metric")
    assert (r.loc[["SW", "CH", "AVI", "MQ", "ANUI"], "z"] > 3).all()
    assert r.loc["S_Dbw", "z"] > 0 and r.loc["DBI", "z"] > 0  # ориентированные z: «лучше случайного» → положительные


def test_configuration_model_null_runs():
    X, A, lab = blobs_and_graph()
    r = evaluate.configuration_model_null(A, lab, n_rep=5, seed=0).set_index("metric")
    assert r.loc["Q", "z"] > 2 and np.isfinite(r["null_mean"]).all()


def test_paired_bootstrap_and_win_rate():
    rng = np.random.default_rng(0)
    vals = pd.DataFrame({"good": rng.normal(0.5, 0.05, 24), "bad": rng.normal(0.3, 0.05, 24), "mid": rng.normal(0.4, 0.05, 24)})
    r = evaluate.paired_bootstrap(vals, n_boot=500, seed=0).set_index(["A", "B"])
    assert r.loc[("good", "bad"), "significant"] and r.loc[("good", "bad"), "ci_lo"] > 0
    w = evaluate.bootstrap_win_rate(vals, n_boot=500, seed=0)
    assert w["good"] > 0.95


def test_borda_and_threshold_orderings():
    table = pd.DataFrame({
        "SW": [0.5, 0.4, 0.3, 0.2], "CH": [500, 400, 300, 200], "S_Dbw": [0.5, 0.6, 0.7, 0.8],
        "AVI": [0.8, 0.7, 0.6, 0.5], "AVU": [0.3, 0.4, 0.5, 0.6], "MQ": [0.6, 0.5, 0.4, 0.3],
        "ANUI": [0.7, 0.6, 0.5, 0.4], "DBI": [0.5, 0.6, 0.7, 0.8], "CH_N": [5, 4, 3, 2], "Qdens": [1, 0.9, 0.8, 0.7],
    }, index=["m1", "m2", "m3", "m4"])
    b = evaluate.borda(table)
    assert list(b.index) == ["m1", "m2", "m3", "m4"]
    t = evaluate.threshold_aggregation(table, grades=2)
    assert t.loc["m1", "rank"] < t.loc["m4", "rank"]
    # компенсация не работает: метод с одной провальной оценкой проигрывает ровному
    table2 = table.copy(); table2.loc["m1", "AVU"] = 0.99
    t2 = evaluate.threshold_aggregation(table2, grades=2)
    assert t2.loc["m1", "n1"] >= 1
