"""Тесты методов кластеризации и выравнивания меток на синтетической SBM с атрибутами (< 60 с)."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score as ari

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from pulsar import align, methods  # noqa: E402


def make_attributed_sbm(n_per: int = 60, k: int = 4, p_in: float = 0.3, p_out: float = 0.02, sep: float = 3.0,
                        seed: int = 0, n_features: int = 5, noise_X: bool = False, noise_A: bool = False):
    """SBM с атрибутами: k блоков по n_per узлов; X — гауссовы облака вокруг центров на расстоянии sep
    (z-стандартизованы); A — симметричная 0/1 csr без диагонали. noise_X / noise_A — заменить источник шумом."""
    rng = np.random.default_rng(seed)
    n = n_per * k
    truth = np.repeat(np.arange(k), n_per)
    F = max(n_features, k)
    centers = np.zeros((k, F))
    centers[np.arange(k), np.arange(k)] = sep
    X = rng.normal(size=(n, F)) + (0 if noise_X else centers[truth])
    X = (X - X.mean(0)) / X.std(0)
    same = truth[:, None] == truth[None, :]
    prob = np.where(same, p_in, p_out) if not noise_A else np.full((n, n), (p_in + (k - 1) * p_out) / k)
    U = np.triu(rng.random((n, n)) < prob, k=1).astype(float)
    A = sp.csr_matrix(U + U.T)
    return X, A, truth


@pytest.fixture(scope="module")
def sbm():
    return make_attributed_sbm(seed=1)


@pytest.mark.parametrize("name", methods.METHODS)
def test_each_method_recovers_truth(sbm, name):
    X, A, truth = sbm
    labels = methods.run_method(name, X, A, k=4, seed=0)
    assert labels.shape == truth.shape and labels.min() == 0
    assert ari(truth, labels) > 0.8, name


def test_kefrin_features_only_ignores_network():
    X, A, truth = make_attributed_sbm(seed=2, noise_A=True)
    res = methods.kefrin(X, A, k=4, rho=1.0, xi=0.0, seed=0)
    assert ari(truth, res.labels) > 0.8
    assert res.centers_P is None


def test_kefrin_network_only_ignores_features():
    X, A, truth = make_attributed_sbm(seed=3, noise_X=True)
    res = methods.kefrin(X, A, k=4, rho=0.0, xi=1.0, seed=0)
    assert ari(truth, res.labels) > 0.8
    assert res.centers_X is None


def test_kefrin_objective_monotone_and_consistent(sbm):
    X, A, truth = sbm
    res = methods.kefrin(X, A, k=4, rho=1.0, xi=1.0, seed=0, n_init=3)
    h = np.asarray(res.history)
    assert len(h) >= 2
    assert np.all(np.diff(h) <= 1e-9 * np.abs(h[:-1]))      # F не растёт по итерациям
    assert res.objective == pytest.approx(methods.kefrin_objective(X, A, res.labels, 1.0, 1.0), rel=1e-9)
    # истинное разбиение должно быть не хуже случайного по F
    rng = np.random.default_rng(0)
    assert methods.kefrin_objective(X, A, truth) < methods.kefrin_objective(X, A, rng.permutation(truth))


def test_kefrin_cosine_runs(sbm):
    X, A, truth = sbm
    res = methods.kefrin(X, A, k=4, metric="cosine", seed=0, n_init=2)
    assert ari(truth, res.labels) > 0.8


def test_leiden_resolution_for_k(sbm):
    X, A, truth = sbm
    for k in (3, 4):
        gamma = methods.leiden_resolution_for_k(A, k, min_size=5, seed=0)
        memb = methods._leiden(methods._to_igraph(A), gamma, seed=0, n_iterations=-1)
        assert int((np.bincount(memb) >= 5).sum()) == k
    params = {"min_size": 5}
    labels = methods.run_leiden(A, 4, seed=0, params=params)
    assert params["k_achieved_"] == 4 and params["gamma_"] > 0
    assert labels.max() == 3


def test_merge_small_clusters():
    A = sp.csr_matrix(np.array([[0, 1, 1, 0, 0, 0],
                                [1, 0, 1, 0, 0, 0],
                                [1, 1, 0, 0, 0, 1],
                                [0, 0, 0, 0, 1, 1],
                                [0, 0, 0, 1, 0, 1],
                                [0, 0, 1, 1, 1, 0]], float))
    labels = np.array([0, 0, 0, 1, 1, 2])          # узел 5 — одиночка, сильнее связан с кластером 1
    out = methods.merge_small_clusters(A, labels, min_size=2)
    assert out.tolist() == [0, 0, 0, 1, 1, 1]


def test_temporal_leiden_same_sbm_three_slices():
    slices = [make_attributed_sbm(seed=s) for s in (10, 11, 12)]
    truth = slices[0][2]
    A_list = [s[1] for s in slices]
    params = {}
    labs = methods.run_temporal_leiden(A_list, k=4, interslice_weight=0.5, seed=0, params=params)
    assert len(labs) == 3 and params["k_achieved_"] == 4
    for lab in labs:
        assert np.unique(lab).size == 4
        assert ari(truth, lab) > 0.95
    for a, b in zip(labs[:-1], labs[1:]):
        assert ari(a, b) > 0.95
        assert (a == b).mean() > 0.95                   # метки сквозные: совпадают поимённо, не только по ARI


def test_align_labels_recovers_permutation():
    rng = np.random.default_rng(0)
    truth = np.repeat(np.arange(5), 20)
    perm = rng.permutation(5)
    cur = perm[truth]
    assert np.array_equal(align.align_labels(truth, cur), truth)
    assert np.array_equal(align.align_to_reference(truth, cur), truth)
    J = align.jaccard_matrix(truth, cur)
    assert J.shape == (5, 5) and np.allclose(J.max(axis=1), 1.0)


def test_align_sequence_new_cluster_gets_fresh_id():
    truth = np.repeat(np.arange(3), 20)
    t1 = truth.copy()
    t2 = truth.copy(); t2[:10] = 7                      # половина кластера 0 отделилась — новый кластер
    t3 = truth[::-1].copy()                             # перестановка номеров
    out = align.align_sequence([t1, t2, t3])
    assert np.array_equal(out[0], t1)
    assert set(np.unique(out[1])) == {0, 1, 2, 3} and out[1][0] == 3 and out[1][15] == 0
    assert np.array_equal(out[2], np.where(t3 == 2, 0, np.where(t3 == 0, 2, 1)))


@pytest.mark.parametrize("name", methods.METHODS)
def test_deterministic_with_same_seed(sbm, name):
    X, A, _ = sbm
    a = methods.run_method(name, X, A, k=4, seed=7)
    b = methods.run_method(name, X, A, k=4, seed=7)
    assert np.array_equal(a, b)


def test_run_all_with_config(sbm):
    X, A, truth = sbm
    cfg = {"methods": ["kmeans", "kefrin"], "params": {"kefrin": {"n_init": 2, "rho": 1.0, "xi": 1.0}}}
    out = methods.run_all(X, A, k=4, seed=0, cfg=cfg)
    assert set(out) == {"kmeans", "kefrin"}
    assert all(v.shape == truth.shape for v in out.values())
