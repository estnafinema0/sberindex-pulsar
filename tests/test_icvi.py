"""Тесты ICVI: граничные случаи, аналитические свойства, сверка с networkx/sklearn и (если доступна) с Pattern."""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from pulsar import icvi  # noqa: E402


def cliques(k: int, size: int, weight: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    n = k * size
    labels = np.repeat(np.arange(k), size)
    A = (labels[:, None] == labels[None, :]).astype(float) * weight
    np.fill_diagonal(A, 0.0)
    return A, labels


def random_graph(n: int, p: float, seed: int, weighted: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    U = np.triu((rng.random((n, n)) < p).astype(float), 1)
    if weighted:
        U *= rng.uniform(0.1, 2.0, size=(n, n))
    return U + U.T


# --------------------------------------------------------------------------- граничные случаи

@pytest.mark.parametrize("k", [2, 3, 4, 7])
def test_isolated_cliques(k):
    A, lab = cliques(k, 6)
    assert icvi.avi(A, lab) == pytest.approx(1.0)
    assert icvi.avu(A, lab) == pytest.approx(0.0)
    assert icvi.anui(A, lab) == pytest.approx(1.0)
    assert icvi.modularity(A, lab) == pytest.approx(1 - 1 / k)
    assert icvi.mq_turbo(A, lab) == pytest.approx(k)


def test_cluster_without_edges_is_penalised():
    A, lab = cliques(3, 5)
    A[lab == 2][:, :] = 0
    A[:, lab == 2] = 0
    A[lab == 2, :] = 0
    assert icvi.avi(A, lab) == pytest.approx(2 / 3)


def test_avu_degenerate_k2_k3():
    rng = np.random.default_rng(1)
    A = random_graph(60, 0.3, 2)
    for k, expected in [(2, 1.0), (3, 2 / 3)]:
        for _ in range(5):
            lab = rng.integers(0, k, size=60)
            if len(np.unique(lab)) < k:
                continue
            assert icvi.avu(A, lab) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("k", [4, 5, 7, 10])
def test_random_partition_expectations(k):
    """У случайного разбиения AVI ≈ 1/K и AVU ≈ (K−1)/(2K−3)."""
    A = random_graph(840, 0.2, 3, weighted=False)
    rng = np.random.default_rng(k)
    avis, avus = [], []
    for _ in range(20):
        lab = rng.permutation(np.repeat(np.arange(k), 840 // k))
        avis.append(icvi.avi(A, lab)); avus.append(icvi.avu(A, lab))
    exp = icvi.expected_random(k)
    assert np.mean(avis) == pytest.approx(exp["AVI"], abs=0.01)
    assert np.mean(avus) == pytest.approx(exp["AVU"], abs=0.01)


def test_avu_sees_only_cut_shape():
    """AVU инвариантен к масштабу внутренних весов: 99 % рёбер внутри и 10 % внутри дают одинаковый AVU."""
    rng = np.random.default_rng(5)
    lab = rng.integers(0, 4, size=80)
    A = random_graph(80, 0.3, 6)
    inside = lab[:, None] == lab[None, :]
    A_strong = np.where(inside, A * 100, A)
    assert icvi.avu(A, lab) == pytest.approx(icvi.avu(A_strong, lab))
    assert icvi.avi(A_strong, lab) > icvi.avi(A, lab)


# --------------------------------------------------------------------------- сверка с эталонами

def test_modularity_matches_networkx():
    A = random_graph(120, 0.1, 7)
    lab = np.random.default_rng(8).integers(0, 5, size=120)
    G = nx.from_numpy_array(A)
    comms = [set(np.flatnonzero(lab == c)) for c in np.unique(lab)]
    assert icvi.modularity(A, lab) == pytest.approx(nx.community.modularity(G, comms, weight="weight"), abs=1e-10)


def test_turbo_mq_equals_k_avi():
    A = random_graph(150, 0.1, 9)
    lab = np.random.default_rng(10).integers(0, 6, size=150)
    k = len(np.unique(lab))
    assert icvi.mq_turbo(A, lab) == pytest.approx(k * icvi.avi(A, lab), abs=1e-10)


def test_sparse_and_dense_agree():
    import scipy.sparse as sp
    A = random_graph(90, 0.2, 11)
    lab = np.random.default_rng(12).integers(0, 4, size=90)
    for f in (icvi.avi, icvi.avu, icvi.modularity, icvi.density_modularity, icvi.mq_basic):
        assert f(A, lab) == pytest.approx(f(sp.csr_matrix(A), lab), abs=1e-12)


def test_labels_arbitrary_values():
    A, lab = cliques(4, 5)
    lab2 = np.array([10, 10, 10, 10, 10, -3, -3, -3, -3, -3, 7, 7, 7, 7, 7, 100, 100, 100, 100, 100])
    assert icvi.graph_metrics(A, lab) == pytest.approx(icvi.graph_metrics(A, lab2))


def test_s_dbw_well_separated_lower_than_mixed():
    rng = np.random.default_rng(13)
    X = np.vstack([rng.normal(c, 0.3, size=(50, 3)) for c in (0, 5, 10, 15)])
    good = np.repeat(np.arange(4), 50)
    bad = rng.permutation(good)
    assert icvi.s_dbw(X, good) < icvi.s_dbw(X, bad)


def test_perm_null_z_positive_for_cliques():
    A, lab = cliques(5, 8)
    r = icvi.perm_null(lambda l: icvi.avi(A, l), lab, n_perm=100, seed=0)
    assert r["z"] > 5 and r["p_greater"] < 0.02
    assert r["null_mean"] == pytest.approx(0.2, abs=0.03)


# --------------------------------------------------------------------------- сверка с Pattern (GPL-3; не вендорим, только импортируем для теста)

PATTERN_DIR = os.environ.get("PATTERN_DIR", "")


def _load_pattern():
    p = pathlib.Path(PATTERN_DIR) / "metrics" / "clustering_metrics.py"
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location("pattern_metrics", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not PATTERN_DIR, reason="PATTERN_DIR не задан")
def test_matches_pattern_library():
    mod = _load_pattern()
    if mod is None:
        pytest.skip("Pattern не найден")
    pm = mod.AdjacencyClusteringMetrics()
    rng = np.random.default_rng(21)
    for trial in range(5):
        A = random_graph(70, 0.25, 30 + trial)
        lab = rng.integers(0, 2 + trial, size=70)
        ref = pm.get_metric(A, lab)
        ours = icvi.graph_metrics(A, lab)
        assert ours["AVI"] == pytest.approx(ref["AVI"], rel=1e-9)
        assert ours["AVU"] == pytest.approx(ref["AVU"], rel=1e-9)
        assert ours["ANUI"] == pytest.approx(ref["ANUI"], rel=1e-9)
        assert ours["Q"] == pytest.approx(ref["modularity"], rel=1e-9)
        assert ours["Qdens"] == pytest.approx(ref["density_modularity"], rel=1e-9)


@pytest.mark.skipif(importlib.util.find_spec("s_dbw") is None, reason="пакет s-dbw не установлен")
def test_s_dbw_matches_package():
    from s_dbw import S_Dbw
    rng = np.random.default_rng(14)
    X = np.vstack([rng.normal(c, 1.0, size=(40, 4)) for c in (0, 3, 6)])
    lab = np.repeat(np.arange(3), 40)
    ref = S_Dbw(X, lab, centers_id=None, method="Halkidi", alg_noise="bind", centr="mean", nearest_centr=False, metric="euclidean")
    assert icvi.s_dbw(X, lab, variant="package") == pytest.approx(ref, rel=1e-9)
    # вариант по статье отличается от пакета (дисперсии vs СКО, плотность по объединению), но упорядочивает так же
    good, bad = lab, rng.permutation(lab)
    assert icvi.s_dbw(X, good) < icvi.s_dbw(X, bad) and icvi.s_dbw(X, good, "package") < icvi.s_dbw(X, bad, "package")
