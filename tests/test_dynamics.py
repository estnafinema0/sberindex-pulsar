"""Тесты динамики типов на синтетике (< 20 с)."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from pulsar import dynamics as dyn  # noqa: E402

MONTHS = [f"{y}-{m:02d}" for y in (2023, 2024) for m in range(1, 13)]


def kmeans_fn(X, A, k, seed):
    return KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(X)


# ----------------------------------------------------------------------------- переходы и Шоррокс

def test_transition_rows_sum_to_one():
    rng = np.random.default_rng(0)
    L = rng.integers(0, 4, size=(24, 300))
    P, C = dyn.transition_matrix(L)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert C.to_numpy().sum() == 23 * 300
    P12, _ = dyn.transition_matrix(L, lag=12)
    assert np.allclose(P12.sum(axis=1), 1.0)


def test_shorrocks_bounds():
    K = 5
    assert dyn.shorrocks(np.eye(K)) == pytest.approx(0.0)
    assert dyn.shorrocks(np.full((K, K), 1 / K)) == pytest.approx(1.0)
    L = np.tile(np.arange(K).repeat(10), (24, 1))           # никто не меняет тип
    assert dyn.shorrocks(dyn.transition_matrix(L)[0]) == pytest.approx(0.0)
    assert dyn.stability_share(L).iloc[-1] == 1.0
    assert dyn.n_changes(L).sum() == 0


# ----------------------------------------------------------------------------- подтверждённые смены

def test_confirmed_transitions_outlier_and_switch():
    T = 12
    L = np.zeros((T, 3), dtype=int)
    L[5, 0] = 1                                              # МО 0: выброс на 1 месяц
    L[6:, 1] = 2                                             # МО 1: смена 0 → 2 с месяца 6 навсегда
    L[10:, 2] = 3                                            # МО 2: хвост длиной 2 < 3 — кандидат
    Lc, ev = dyn.confirmed_transitions(L, min_run=3, months=MONTHS[:T])
    assert (Lc[:, 0] == 0).all()
    assert (Lc[:6, 1] == 0).all() and (Lc[6:, 1] == 2).all()
    assert (Lc[:, 2] == 0).all()
    conf = ev[ev.status == "confirmed"]
    assert len(conf) == 1
    r = conf.iloc[0]
    assert (r.territory_index, r.t, r.month, r["from"], r["to"]) == (1, 6, "2023-07", 0, 2)
    assert r.confirmed_t == 8
    cand = ev[ev.status == "candidate"]
    assert list(cand.territory_index) == [2] and cand.iloc[0].month == "2023-11"


def test_confirmed_initial_and_return():
    # 0 1 1 1 2 2 1 1 1 1: начальный тип 1 (первый отрезок ≥3), отлучка в 2 на 2 мес. не засчитана
    x = np.array([0, 1, 1, 1, 2, 2, 1, 1, 1, 1])
    Lc, ev = dyn.confirmed_transitions(x[:, None], min_run=3)
    assert (Lc[:, 0] == 1).all() and ev.empty


def test_seasonal_decomposition_partition():
    rng = np.random.default_rng(1)
    L = np.repeat(rng.integers(0, 3, size=(1, 200)), 24, axis=0)
    L[12, :50] = (L[12, :50] + 1) % 3                       # шум в январе 2024 (декабрь → январь)
    L[5:, 50:80] = (L[5:, 50:80] + 1) % 3                   # подтверждённая смена в июне 2023
    Lc, ev = dyn.confirmed_transitions(L, months=MONTHS)
    dec = dyn.seasonal_decomposition_of_changes(L, Lc, MONTHS)
    tot = dec.iloc[-1]
    parts = tot.n_confirmed + tot.n_noise_short + tot.n_noise_return + tot.n_candidate
    assert parts == tot.n_raw == 50 * 2 + 30
    assert tot.n_confirmed == 30
    ny = dec[dec.is_new_year].iloc[0]
    assert ny.n_raw == 50 and ny.n_noise_short == 50
    cs = dyn.calendar_share(ev, MONTHS)
    assert cs.loc[6, "share"] == 1.0 and cs["share"].sum() == pytest.approx(1.0)


# ----------------------------------------------------------------------------- Greene

def test_greene_merge_split_continue():
    a = np.repeat([0, 1, 2], 30)
    merged = np.where(a == 1, 0, a)                          # 0 и 1 сливаются
    ev = dyn.greene_events(np.vstack([a, merged]))
    assert (ev.event == "merge").sum() == 1
    m = ev[ev.event == "merge"].iloc[0]
    assert sorted(m.from_ids) == [0, 1] and m.to_ids == [0]
    ev = dyn.greene_events(np.vstack([merged, a]))
    assert (ev.event == "split").sum() == 1
    perm = np.array([2, 0, 1])[a]                            # только перенумерация
    ev = dyn.greene_events(np.vstack([a, perm]))
    assert set(ev.event) == {"continue"} and len(ev) == 3
    assert np.allclose(ev.jaccard, 1.0)
    sizes = dyn.cluster_sizes_over_time(np.vstack([a, merged]))
    assert sizes.loc["1", 0] == 60 and sizes.loc["1", 1] == 0


# ----------------------------------------------------------------------------- бутстрэп

def test_bootstrap_confidence_center_vs_border():
    rng = np.random.default_rng(0)
    n = 150
    left, right = rng.normal([-3, 0], 0.7, (n, 2)), rng.normal([3, 0], 0.7, (n, 2))
    mid = (left[:, 0].mean() + right[:, 0].mean()) / 2      # точная середина между эмпирическими центрами
    X = np.vstack([left, right, np.column_stack([np.full(10, mid), rng.normal(0, 0.1, 10)])])
    prob, conf = dyn.bootstrap_membership(X, None, None, 2, n_boot=30, frac=0.8, seed=0, run_fn=kmeans_fn)
    assert prob.shape == (X.shape[0], 2)
    assert np.allclose(prob.sum(axis=1), 1.0)
    center = np.abs(X[:2 * n, 0] - np.sign(X[:2 * n, 0]) * 3) < 1.0
    assert conf[:2 * n][center].min() > 0.9
    assert conf[2 * n:].mean() < conf[:2 * n][center].mean()
    assert conf[2 * n:].mean() < 0.9


# ----------------------------------------------------------------------------- out-of-time

def make_panel(shift: bool, seed: int = 0, n_per: int = 60, k: int = 3, T: int = 24):
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(k), n_per)
    centers = np.eye(k) * 4.0
    L = np.tile(truth, (T, 1))
    if shift:
        L[18:, :10] = 1                                      # 10 МО типа 0 переходят в тип 1 с 2024-07
    X = centers[L] + rng.normal(0, 0.6, (T, truth.size, k))
    return X, L


def test_out_of_time_stationary():
    X, L = make_panel(shift=False)
    res = dyn.out_of_time(X, L, np.arange(12), np.arange(12, 24), months=MONTHS)
    assert res["pred"].shape == (12, L.shape[1])
    assert res["ari_mean"] > 0.9 and res["agreement_mean"] > 0.95
    assert res["margin_last"].shape == (L.shape[1],)
    assert res["shorrocks_last"] < 0.1


def test_out_of_time_detects_shift():
    X, L = make_panel(shift=True)
    res = dyn.out_of_time(X, L, np.arange(12), np.arange(12, 24), months=MONTHS)
    assert (res["pred"][-1, :10] == 1).mean() > 0.9
    assert res["confirmed"]["recall"] >= 0.9
    ev = res["events_online"]
    found = ev[(ev.status == "confirmed") & (ev.territory_index < 10)]
    assert (found.month == "2024-07").mean() > 0.8
    assert res["transition_last_counts"].loc[0, 1] >= 9


def test_centroid_classifier_and_margin():
    C = np.array([[0.0, 0.0], [10.0, 0.0]])
    Xt = np.array([[1.0, 0.0], [5.0, 0.0], [9.0, 0.0]])
    assert list(dyn.centroid_classifier(C, Xt)[[0, 2]]) == [0, 1]   # середина (5, 0) — ничья
    m = dyn.margin(C, Xt)
    assert m[0] == pytest.approx(8.0) and m[1] == pytest.approx(0.0)


# ----------------------------------------------------------------------------- таблицы и сводка

def test_trajectory_table_counts():
    L = np.zeros((6, 3), dtype=int)
    L[3:, 1] = 2
    L[2:, 2] = 1
    L[4:, 2] = 3
    meta = pd.DataFrame({"territory_id": ["a", "b", "c"], "name": ["A", "B", "C"]})
    tt = dyn.trajectory_table(L, MONTHS[:6], meta)
    assert list(tt.n_changes) == [0, 1, 2]
    assert pd.isna(tt.last_change_month[0]) and list(tt.last_change_month[1:]) == ["2023-04", "2023-05"]
    assert tt.loc[2, "trajectory"] == "0,0,1,1,3,3"
    assert list(tt.last_type) == [0, 2, 3] and "name" in tt.columns


def test_summarize_json():
    rng = np.random.default_rng(3)
    L = np.repeat(rng.integers(0, 4, size=(1, 300)), 24, axis=0)
    flip = rng.random(L.shape) < 0.05
    L = np.where(flip, rng.integers(0, 4, size=L.shape), L)
    Lc, _ = dyn.confirmed_transitions(L, months=MONTHS)
    s = dyn.summarize(L, Lc, MONTHS)
    txt = json.dumps(s, ensure_ascii=False)
    assert "shorrocks" in s and s["K"] == 4
    assert s["shorrocks"]["confirmed"]["lag1"] <= s["shorrocks"]["raw"]["lag1"]
    assert len(s["sizes"]["0"]) == 24 and len(txt) > 0
