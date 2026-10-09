"""Тесты слоёв графа близости и SNF на синтетике: инварианты, разделимость, со-движение, метрики."""
from __future__ import annotations

import pathlib
import sys

import igraph as ig
import numpy as np
import pytest
import scipy.sparse as sp
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from pulsar import graphs  # noqa: E402

CFG_PATH = pathlib.Path(__file__).resolve().parents[1] / "configs" / "graph.yaml"


# --------------------------------------------------------------------------- синтетика

def three_clouds(n_per: int = 60, dim: int = 7, sep: float = 12.0, seed: int = 0):
    """Три хорошо разделённых гауссовых облака."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((3, dim))
    centers[1, 0] = sep
    centers[2, 1] = sep
    labels = np.repeat(np.arange(3), n_per)
    X = centers[labels] + rng.normal(size=(3 * n_per, dim))
    return X, labels


def block_series(W: int = 24, n_per: int = 40, seed: int = 1):
    """Два блока МО с общим фактором в каждом; строки — месяцы, столбцы — МО."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(W, 2))
    labels = np.repeat(np.arange(2), n_per)
    G = f[:, labels] + 0.6 * rng.normal(size=(W, 2 * n_per))
    return G, labels


def ring_distances(n: int = 50, step_km: float = 30.0):
    """Точки на окружности: дорожные расстояния вдоль кольца."""
    i = np.arange(n)
    d = np.abs(i[:, None] - i[None, :])
    d = np.minimum(d, n - d) * step_km
    return d.astype(float)


def inter_share(A: sp.csr_matrix, labels: np.ndarray) -> float:
    U = sp.triu(A, k=1).tocoo()
    return float((labels[U.row] != labels[U.col]).mean())


def modularity(A: sp.csr_matrix, labels: np.ndarray) -> float:
    U = sp.triu(A, k=1).tocoo()
    g = ig.Graph(n=A.shape[0], edges=list(zip(U.row.tolist(), U.col.tolist())))
    return float(g.modularity(labels.tolist(), weights=U.data.tolist()))


def check_invariants(A: sp.csr_matrix, n: int):
    assert sp.isspmatrix_csr(A) and A.shape == (n, n)
    assert abs(A - A.T).max() == 0 if A.nnz else True
    assert A.diagonal().sum() == 0
    if A.nnz:
        assert A.data.min() > 0 and A.data.max() <= 1.0 + 1e-12


# --------------------------------------------------------------------------- инварианты

@pytest.fixture(scope="module")
def clouds():
    return three_clouds()


@pytest.fixture(scope="module")
def layers(clouds):
    X, _ = clouds
    n = X.shape[0]
    G, _ = block_series(n_per=n // 2)
    D = ring_distances(n)
    return {
        "behaviour": graphs.behaviour_layer(X),
        "comovement": graphs.comovement_layer(G),
        "gravity": graphs.gravity_layer(D),
    }


def test_layers_invariants(layers, clouds):
    n = clouds[0].shape[0]
    for A in layers.values():
        check_invariants(A, n)
        assert A.nnz > 0


def test_behaviour_connected_without_isolates(clouds):
    X, _ = clouds
    A = graphs.behaviour_layer(X, k=5, add_mst=True)
    U = sp.triu(A, k=1).tocoo()
    g = ig.Graph(n=X.shape[0], edges=list(zip(U.row.tolist(), U.col.tolist())))
    assert len(g.connected_components()) == 1
    assert min(g.degree()) >= 1
    # без MST три облака — три компоненты
    B = graphs.behaviour_layer(X, k=5, add_mst=False)
    Ub = sp.triu(B, k=1).tocoo()
    gb = ig.Graph(n=X.shape[0], edges=list(zip(Ub.row.tolist(), Ub.col.tolist())))
    assert len(gb.connected_components()) >= 3


def test_behaviour_separates_clouds(clouds):
    X, labels = clouds
    A = graphs.behaviour_layer(X)
    assert inter_share(A, labels) < 0.05
    assert modularity(A, labels) > 0.5


def test_behaviour_mutual_sparser_than_union(clouds):
    X, _ = clouds
    assert graphs.behaviour_layer(X, mutual=True, add_mst=False).nnz <= \
        graphs.behaviour_layer(X, mutual=False, add_mst=False).nnz


# --------------------------------------------------------------------------- со-движение

def test_comovement_blocks():
    G, labels = block_series()
    A = graphs.comovement_layer(G, k=5)
    check_invariants(A, G.shape[1])
    assert inter_share(A, labels) < 0.1


def test_comovement_lag_detects_shift():
    rng = np.random.default_rng(3)
    W, n = 24, 30
    base = rng.normal(size=W + 1)
    G = rng.normal(size=(W, n))
    G[:, 0] = base[1:]          # ряд 0
    G[:, 1] = base[:-1]         # ряд 1 — тот же ряд, сдвинутый на месяц
    A1 = graphs.comovement_layer(G, k=3, lags=(0, 1), shrinkage="none")
    A0 = graphs.comovement_layer(G, k=3, lags=(0,), shrinkage="none")
    assert A1[0, 1] > 0.9
    assert A0[0, 1] < 0.5
    # чистый шум + одна пара: LW даёт δ = 1 («структуры нет») → слой пустой по построению
    assert graphs.comovement_shrinkage(G)[0] == pytest.approx(1.0)
    assert graphs.comovement_layer(G, k=3, lags=(0, 1)).nnz == 0


def test_comovement_lag_with_shrinkage():
    """При реальной структуре (блоки) δ < 1, и сдвинутая копия ряда — сильнейший сосед с лагом 1."""
    G, labels = block_series(W=24, n_per=40, seed=7)
    G[1:, 1] = G[:-1, 0]        # ряд 1 = ряд 0, запаздывающий на месяц
    G[0, 1] = 0.0
    deltas = graphs.comovement_shrinkage(G)
    assert 0 < deltas[0] < 1 and 0 < deltas[1] < 1
    A1 = graphs.comovement_layer(G, k=5, lags=(0, 1))
    A0 = graphs.comovement_layer(G, k=5, lags=(0,))
    row1 = A1[0].toarray().ravel()
    assert row1.argmax() == 1 and row1[1] > 0
    assert A1[0, 1] > A0[0, 1]


def test_comovement_empty_cases():
    assert graphs.comovement_layer(None).shape == (0, 0)
    E = graphs.comovement_layer(np.zeros((2, 10)))
    assert E.shape == (10, 10) and E.nnz == 0


def test_comovement_shrinkage_shrinks():
    G, _ = block_series()
    Z = graphs._standardize_columns(G)
    R, delta = graphs._lw_correlation(Z)
    assert 0 < delta < 1
    assert np.allclose(np.diag(R), 1.0)
    assert np.abs(R - np.eye(len(R))).max() < np.abs(Z.T @ Z / len(Z) - np.eye(len(R))).max()


# --------------------------------------------------------------------------- география

def test_gravity_nearest_strongest():
    D = ring_distances(50)
    A = graphs.gravity_layer(D, k=4, scale_km=100.0)
    check_invariants(A, 50)
    for i in range(50):
        row = A[i].toarray().ravel()
        nbrs = np.flatnonzero(row)
        assert len(nbrs) >= 4
        assert set(nbrs) >= {(i - 1) % 50, (i + 1) % 50}
        assert row[(i + 1) % 50] == pytest.approx(np.exp(-30.0 / 100.0))
        assert row.max() == row[(i + 1) % 50]


def test_gravity_zero_distance_replaced():
    D = ring_distances(20)
    D[0, 1] = D[1, 0] = 0.0
    cfg = yaml.safe_load(CFG_PATH.read_text())
    X = np.random.default_rng(0).normal(size=(20, 3))
    out = graphs.build_month(X, None, D, cfg)
    assert out["gravity"][0, 1] == pytest.approx(np.exp(-cfg["gravity"]["zero_distance_km"] / cfg["gravity"]["scale_km"]))


# --------------------------------------------------------------------------- SNF

def test_snf_identity_layers(clouds):
    X, labels = clouds
    A = graphs.behaviour_layer(X, k=10)
    F = graphs.snf_fuse([A, A.copy()], k=10, t=10)
    check_invariants(F, X.shape[0])
    assert graphs.edge_jaccard(F, A) > 0.6
    assert inter_share(F, labels) < 0.05
    # связность и отсутствие изолятов у слитого графа (MST)
    U = sp.triu(F, k=1).tocoo()
    g = ig.Graph(n=X.shape[0], edges=list(zip(U.row.tolist(), U.col.tolist())))
    assert len(g.connected_components()) == 1 and min(g.degree()) >= 1


def test_snf_skips_empty_and_single(clouds):
    X, _ = clouds
    n = X.shape[0]
    A = graphs.behaviour_layer(X)
    empty = sp.csr_matrix((n, n))
    F = graphs.snf_fuse([A, empty], k=10)
    assert F.shape == (n, n) and F.nnz > 0
    assert graphs.snf_fuse([empty, empty]).nnz == 0


def test_snf_combines_views(clouds):
    """Шумный второй слой не должен разрушать структуру первого."""
    X, labels = clouds
    rng = np.random.default_rng(5)
    A = graphs.behaviour_layer(X)
    noise = graphs.behaviour_layer(rng.normal(size=X.shape))
    F = graphs.snf_fuse([A, noise], k=10, t=10)
    assert modularity(F, labels) > 0.3


def test_build_month_keys(clouds):
    X, _ = clouds
    n = X.shape[0]
    cfg = yaml.safe_load(CFG_PATH.read_text())
    G, _ = block_series(W=18, n_per=n // 2)
    out = graphs.build_month(X, G, ring_distances(n), cfg)
    assert set(out) == {"behaviour", "comovement", "gravity", "fused"}
    for A in out.values():
        check_invariants(A, n)
    assert cfg["primary"] in out
    # короткая история → пустой слой со-движения, остальное работает
    out2 = graphs.build_month(X, G[:2], ring_distances(n), cfg)
    assert out2["comovement"].nnz == 0 and out2["fused"].nnz > 0
    cfg["fusion"]["method"] = "mean"
    assert graphs.build_month(X, G, ring_distances(n), cfg)["fused"].nnz > 0


# --------------------------------------------------------------------------- метрики и утилиты

def test_network_summary_keys(clouds):
    X, labels = clouds
    A = graphs.behaviour_layer(X)
    s = graphs.network_summary(A, X)
    keys = {"n_nodes", "n_edges", "density", "mean_degree", "n_components", "isolate_share",
            "degree_assortativity", "algebraic_connectivity", "spectral_gap", "spectral_gap_k",
            "avg_clustering", "mean_edge_length", "rayleigh_quotient", "rayleigh_quotient_norm"}
    assert keys <= set(s)
    assert s["n_nodes"] == X.shape[0] and s["n_edges"] == sp.triu(A, 1).nnz
    assert 0 < s["density"] < 0.2
    assert s["n_components"] == 1 and s["isolate_share"] == 0
    assert 0 < s["algebraic_connectivity"] < 1
    assert s["spectral_gap_k"] == 3                      # три облака → зазор после третьего с.з.
    assert 0 <= s["avg_clustering"] <= 1
    assert -1 <= s["degree_assortativity"] <= 1
    assert s["mean_edge_length"] > 0
    assert 0 <= s["rayleigh_quotient_norm"] <= 2
    # случайные признаки — грубее на графе, чем те, по которым он строился
    Xr = np.random.default_rng(0).normal(size=X.shape)
    assert graphs.network_summary(A, Xr)["rayleigh_quotient"] > s["rayleigh_quotient"]


def test_network_summary_empty():
    s = graphs.network_summary(sp.csr_matrix((5, 5)))
    assert s["n_edges"] == 0 and s["isolate_share"] == 1.0 and s["n_components"] == 5


def test_edge_jaccard():
    A = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float))
    B = sp.csr_matrix(np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], float))
    assert graphs.edge_jaccard(A, A) == 1.0
    assert graphs.edge_jaccard(A, B) == 0.0
    assert graphs.edge_jaccard(A, A + B) == pytest.approx(2 / 3)


def test_save_load_roundtrip(tmp_path, clouds):
    A = graphs.behaviour_layer(clouds[0])
    p = tmp_path / "g.npz"
    graphs.save_graph(A, p)
    assert abs(graphs.load_graph(p) - A).max() == 0
