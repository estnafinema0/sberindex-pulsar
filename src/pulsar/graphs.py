"""Помесячные графы экономической близости МО: три слоя и их слияние.

Слои (все — симметричные csr-матрицы N×N без диагонали, веса в [0, 1]):
* behaviour  — взаимный kNN по признакам месяца с самонастраивающимся гауссовым ядром
               (Zelnik-Manor & Perona, 2004): w_ij = exp(−d_ij² / (σ_i σ_j)), σ_i — расстояние до
               local_scale_k-го соседа; плюс рёбра MST для связности.
* comovement — со-движение: корреляции лог-приростов за окно W с усадкой Ледуа–Вольфа (2004),
               максимум по лагам ±lag, отрицательные → 0, kNN-разрежение по весу.
* gravity    — география: w_ij = exp(−D_ij / scale_km) по дорожным расстояниям, k ближайших.
* fused      — Similarity Network Fusion (Wang et al., Nature Methods 2014), своя реализация.

Все формулы реализованы самостоятельно по публикациям; гиперпараметры — в configs/graph.yaml.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.linalg import eigsh
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

_EPS = 1e-12


# ----------------------------------------------------------------------------- вспомогательное

def _finalize(A: sp.spmatrix, n: int) -> sp.csr_matrix:
    """Привести к csr: без диагонали, без явных нулей, веса обрезаны в [0, 1], симметрия по max."""
    A = sp.csr_matrix(A, shape=(n, n), dtype=np.float64)
    A = A.maximum(A.T)
    A.setdiag(0.0)
    A.eliminate_zeros()
    A.data = np.clip(A.data, 0.0, 1.0)
    A.eliminate_zeros()
    A.sort_indices()
    return A


def _knn_from_dense(S: np.ndarray, k: int, mutual: bool) -> sp.csr_matrix:
    """Оставить в каждой строке плотной матрицы сходства S k наибольших положительных весов.

    mutual=True — ребро остаётся только если оно вошло в top-k обеих вершин (поэлементный min),
    иначе объединение (поэлементный max). Диагональ и неположительные веса не участвуют.
    """
    n = S.shape[0]
    k = int(min(k, n - 1))
    if k <= 0:
        return sp.csr_matrix((n, n))
    S = np.array(S, dtype=np.float64, copy=True)
    np.fill_diagonal(S, -np.inf)
    cols = np.argpartition(-S, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    vals = S[rows, cols.ravel()]
    keep = vals > 0
    W = sp.csr_matrix((vals[keep], (rows[keep], cols.ravel()[keep])), shape=(n, n))
    return W.minimum(W.T) if mutual else W.maximum(W.T)


def _knn_sparse(dist: np.ndarray, idx: np.ndarray, w: np.ndarray, n: int, mutual: bool) -> sp.csr_matrix:
    """Собрать kNN-граф из массивов соседей (n×k) с весами w; взаимность — через min/max."""
    rows = np.repeat(np.arange(n), idx.shape[1])
    W = sp.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n, n))
    return W.minimum(W.T) if mutual else W.maximum(W.T)


def _standardize_columns(G: np.ndarray) -> np.ndarray:
    """Z-нормировка столбцов (рядов МО); константные ряды → нули; NaN → 0 (среднее)."""
    G = np.asarray(G, dtype=np.float64)
    mu = np.nanmean(G, axis=0)
    sd = np.nanstd(G, axis=0)
    sd = np.where(sd > _EPS, sd, 1.0)
    Z = (G - mu) / sd
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def _lw_correlation(Z: np.ndarray) -> tuple[np.ndarray, float]:
    """Корреляции с усадкой Ледуа–Вольфа: Σ_LW = (1−δ)S + δμI → нормировка к корреляциям.

    Z — матрица (T × p) со стандартизованными столбцами; возвращает (R, δ). Для T ≪ p
    выборочная ковариация вырождена, усадка делает оценку хорошо обусловленной.
    """
    T, p = Z.shape
    delta = float(ledoit_wolf_shrinkage(Z, assume_centered=True, block_size=2000))
    S = (Z.T @ Z) / T                                   # выборочная ковариация (данные центрированы)
    mu = float(np.trace(S)) / p
    C = (1.0 - delta) * S
    C[np.diag_indices(p)] += delta * mu                 # Σ_LW = (1−δ)S + δμI; precision не нужна (экономим pinvh)
    d = np.sqrt(np.clip(np.diag(C), _EPS, None))
    R = C / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R, delta


# ----------------------------------------------------------------------------- слои

def behaviour_layer(X: np.ndarray, k: int = 10, mutual: bool = True, local_scale_k: int = 7,
                    add_mst: bool = True) -> sp.csr_matrix:
    """Слой «поведение»: взаимный kNN по евклиду в признаковом пространстве с локальным масштабом.

    w_ij = exp(−d_ij² / (σ_i σ_j)), σ_i = расстояние от i до local_scale_k-го соседа.
    add_mst — добавить рёбра минимального остовного дерева (по евклидовым расстояниям, вес — то же
    ядро), чтобы граф был связным и без изолятов.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        return sp.csr_matrix((n, n))
    k = int(min(k, n - 1))
    local_scale_k = int(min(max(local_scale_k, 1), n - 1))
    m = max(k, local_scale_k)
    nn = NearestNeighbors(n_neighbors=m + 1).fit(X)
    dist, idx = nn.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]                    # без самой точки
    sigma = np.maximum(dist[:, local_scale_k - 1], _EPS)   # локальный масштаб
    d_k, i_k = dist[:, :k], idx[:, :k]
    w = np.exp(-d_k ** 2 / (sigma[:, None] * sigma[i_k]))
    A = _knn_sparse(d_k, i_k, w, n, mutual)
    if add_mst:
        Dfull = pairwise_distances(X)
        np.fill_diagonal(Dfull, 0.0)
        Dfull = Dfull + 1e-9                                # нулевые расстояния ≠ «нет ребра»
        np.fill_diagonal(Dfull, 0.0)
        T = minimum_spanning_tree(Dfull).tocoo()
        wt = np.exp(-(T.data - 1e-9) ** 2 / (sigma[T.row] * sigma[T.col]))
        Tw = sp.csr_matrix((wt, (T.row, T.col)), shape=(n, n))
        A = A.maximum(Tw).maximum(Tw.T)
    return _finalize(A, n)


def _lag_correlations(G: np.ndarray, lags, use_lw: bool):
    """Генератор (lag, C, δ): симметризованная по max матрица корреляций для каждого лага и усадка δ."""
    W, n = G.shape
    for lag in sorted({int(abs(l)) for l in lags}):
        if lag >= W - 2:                                    # меньше 3 перекрывающихся точек — лаг не считаем
            continue
        if lag == 0:
            Z = _standardize_columns(G)
            C, delta = _lw_correlation(Z) if use_lw else ((Z.T @ Z) / W, 0.0)
        else:
            Za = _standardize_columns(G[lag:])              # ряд i «позже»
            Zb = _standardize_columns(G[:-lag])             # ряд j «раньше»: j опережает i на lag
            if use_lw:
                R, delta = _lw_correlation(np.hstack([Za, Zb]))
                C = R[:n, n:]
            else:
                C, delta = (Za.T @ Zb) / (W - lag), 0.0
            C = np.maximum(C, C.T)
        yield lag, C, delta


def comovement_shrinkage(G: np.ndarray, lags=(0, 1)) -> dict:
    """Диагностика: усадка Ледуа–Вольфа δ по лагам (δ → 1 означает «корреляции неотличимы от шума»)."""
    G = np.asarray(G, dtype=np.float64)
    return {lag: delta for lag, _, delta in _lag_correlations(G, lags, use_lw=True)}


def comovement_layer(G: np.ndarray | None, k: int = 10, lags=(0, 1), shrinkage: str = "ledoit_wolf",
                     mutual: bool = True) -> sp.csr_matrix:
    """Слой «со-движение»: корреляции рядов приростов МО по окну W с усадкой Ледуа–Вольфа.

    G — матрица (W × N), строки — месяцы ≤ t. Для лага L>0 кросс-корреляция ряда i, сдвинутого
    на L, с рядом j (и симметрично), усадка — LW на матрице [Z_{L:}, Z_{:−L}] (W−L × 2N).
    Итоговое сходство ρ_ij = max по лагам, отрицательные → 0; затем k сильнейших на вершину.
    Для стандартизованных рядов LW-корреляция = (1−δ)·r_ij, т.е. ранжирование соседей совпадает с
    выборочным, а δ задаёт масштаб весов; при δ = 1 (структуры нет) слой пустой — это осознанно.
    shrinkage="none" — выборочные корреляции без усадки (для абляции).
    Если G None или W < 3 — пустая матрица.
    """
    if G is None:
        return sp.csr_matrix((0, 0))
    G = np.asarray(G, dtype=np.float64)
    W, n = G.shape
    if W < 3 or n < 2:
        return sp.csr_matrix((n, n))
    R = np.zeros((n, n))
    for _, C, _ in _lag_correlations(G, lags, use_lw=(shrinkage == "ledoit_wolf")):
        R = np.maximum(R, np.nan_to_num(C))
    R = np.clip(R, 0.0, 1.0)
    np.fill_diagonal(R, 0.0)
    return _finalize(_knn_from_dense(R, k, mutual), n)


def gravity_layer(D: np.ndarray, k: int = 10, scale_km: float = 200.0, mutual: bool = False) -> sp.csr_matrix:
    """Слой «география»: w_ij = exp(−D_ij / scale_km), k ближайших по дороге, симметризация по max."""
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    S = np.exp(-D / float(scale_km))
    return _finalize(_knn_from_dense(S, k, mutual), n)


# ----------------------------------------------------------------------------- слияние (SNF)

def _snf_full_kernel(W: np.ndarray, alpha: float) -> np.ndarray:
    """P = «полное ядро» SNF: P_ij = W_ij / (2 Σ_{k≠i} W_ik), P_ii = 1/2 (строки суммируются в 1).

    alpha — вес самопетли, добавляемый перед нормировкой (регуляризация диффузии; при alpha=0.5
    и нулевой диагонали W это ровно формула Wang et al.). Вершина без рёбер остаётся на месте (P_ii=1).
    """
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    rs = W.sum(axis=1)
    P = np.zeros_like(W)
    ok = rs > _EPS
    P[ok] = W[ok] * (1.0 - alpha) / rs[ok, None]
    P[np.arange(len(W)), np.arange(len(W))] = np.where(ok, alpha, 1.0)
    return P


def _snf_local_kernel(W: np.ndarray, k: int) -> sp.csr_matrix:
    """S = «локальное ядро» SNF: S_ij = W_ij / Σ_{l∈N_i} W_il для j ∈ N_i, иначе 0.

    Как в статье, окрестность N_i включает саму вершину (k ближайших + i); самоподобие W_ii = 1 —
    максимум шкалы [0, 1]. Без этого диффузия S·P·Sᵀ теряет прямые рёбра и мерит лишь общих соседей.
    """
    n = W.shape[0]
    Wk = np.array(W, dtype=np.float64, copy=True)
    np.fill_diagonal(Wk, -np.inf)
    kk = int(min(k, n - 1))
    cols = np.argpartition(-Wk, kk - 1, axis=1)[:, :kk]
    rows = np.repeat(np.arange(n), kk)
    vals = Wk[rows, cols.ravel()]
    keep = vals > 0
    S = sp.csr_matrix((vals[keep], (rows[keep], cols.ravel()[keep])), shape=(n, n)) + sp.eye(n, format="csr")
    rs = np.asarray(S.sum(axis=1)).ravel()
    return (sp.diags(1.0 / rs) @ S).tocsr()


def _max_spanning_tree(F: np.ndarray) -> sp.csr_matrix:
    """Максимальное остовное дерево плотной матрицы сходства F ≥ 0 (рёбра — положительные F_ij)."""
    n = F.shape[0]
    Dist = np.where(F > 0, 1.0 + 1e-9 - F, 0.0)            # больше сходство → короче «расстояние»
    np.fill_diagonal(Dist, 0.0)
    T = minimum_spanning_tree(Dist).tocoo()
    Tw = sp.csr_matrix((F[T.row, T.col], (T.row, T.col)), shape=(n, n))
    return Tw.maximum(Tw.T)


def _sparsify_fused(F: np.ndarray, k: int, mutual: bool, add_mst: bool) -> sp.csr_matrix:
    """Разрежение слитой плотной матрицы: kNN по весу (+ MST для связности), масштаб к max = 1."""
    n = F.shape[0]
    A = _knn_from_dense(F, k, mutual=mutual)
    if add_mst:
        A = A.maximum(_max_spanning_tree(F))
    if A.nnz:
        A = A / A.data.max()
    return _finalize(A, n)


def snf_fuse(layers: list, k: int = 10, t: int = 10, alpha: float = 0.5, mutual: bool = True,
             add_mst: bool = True) -> sp.csr_matrix:
    """Similarity Network Fusion (Wang et al., 2014) для списка слоёв; пустые слои пропускаются.

    Итерации: P^(v) ← S^(v) · mean_{u≠v} P^(u) · S^(v)ᵀ, затем перенормировка; после t шагов
    P = mean_v P^(v), симметризация, разрежение до k сильнейших на вершину (mutual — взаимно),
    add_mst — рёбра максимального остовного дерева P для связности, масштаб к [0, 1].
    Диффузия идёт по разреженным локальным ядрам S, поэтому шаг стоит O(N² k).
    """
    dense = []
    for L in layers:
        if L is None or L.shape[0] == 0 or L.nnz == 0:
            continue
        dense.append(np.asarray(sp.csr_matrix(L).todense(), dtype=np.float64))
    if not dense:
        n = layers[0].shape[0] if layers else 0
        return sp.csr_matrix((n, n))
    n = dense[0].shape[0]
    if len(dense) == 1:
        return _sparsify_fused(dense[0], k, mutual, add_mst)
    P = [_snf_full_kernel(W, alpha) for W in dense]
    S = [_snf_local_kernel(W, k) for W in dense]
    m = len(P)
    for _ in range(int(t)):
        total = sum(P)
        new = []
        for v in range(m):
            others = (total - P[v]) / (m - 1)
            Pv = np.asarray(S[v] @ others @ S[v].T)         # sparse · dense · sparse → dense
            new.append(_snf_full_kernel((Pv + Pv.T) / 2.0, alpha))
        P = new
    F = sum(P) / m
    F = (F + F.T) / 2.0
    np.fill_diagonal(F, 0.0)
    return _sparsify_fused(F, k, mutual, add_mst)


def mean_fuse(layers: list, k: int = 10, mutual: bool = True, add_mst: bool = True) -> sp.csr_matrix:
    """Базовое слияние для абляции: среднее слоёв (каждый отмасштабирован к max=1), затем kNN (+MST)."""
    dense = [np.asarray(L.todense()) / max(L.data.max(), _EPS) for L in layers if L is not None and L.nnz]
    if not dense:
        n = layers[0].shape[0] if layers else 0
        return sp.csr_matrix((n, n))
    F = sum(dense) / len(dense)
    return _sparsify_fused(F, k, mutual, add_mst)


# ----------------------------------------------------------------------------- сборка месяца

def build_month(X: np.ndarray, G: np.ndarray | None, D: np.ndarray, cfg: dict) -> dict:
    """Построить все слои месяца по конфигу (configs/graph.yaml) и слить их."""
    cb = dict(cfg.get("behaviour", {}))
    cc = dict(cfg.get("comovement", {}))
    cg = dict(cfg.get("gravity", {}))
    cf = dict(cfg.get("fusion", {}))
    n = np.asarray(X).shape[0]

    out = {"behaviour": behaviour_layer(X, k=cb.get("k", 10), mutual=cb.get("mutual", True),
                                        local_scale_k=cb.get("local_scale_k", 7),
                                        add_mst=cb.get("add_mst", True))}

    window, min_window = int(cc.get("window", 12)), int(cc.get("min_window", 3))
    Gw = None
    if G is not None:
        Gw = np.asarray(G)[-window:]
        if Gw.shape[0] < min_window:
            Gw = None
    C = comovement_layer(Gw, k=cc.get("k", 10), lags=tuple(cc.get("lags", (0, 1))),
                         shrinkage=cc.get("shrinkage", "ledoit_wolf"), mutual=cc.get("mutual", True))
    out["comovement"] = C if C.shape[0] == n else sp.csr_matrix((n, n))

    Dm = np.array(D, dtype=np.float64, copy=True)
    zero_km = float(cg.get("zero_distance_km", 0.0))
    if zero_km > 0:                                           # страховка: «нулевые» расстояния между разными МО
        off = ~np.eye(n, dtype=bool)
        Dm[off & (Dm <= 0)] = zero_km
    out["gravity"] = gravity_layer(Dm, k=cg.get("k", 10), scale_km=cg.get("scale_km", 200.0),
                                   mutual=cg.get("mutual", False))

    layers = [out["behaviour"], out["comovement"], out["gravity"]]
    method = cf.get("method", "snf")
    kw = dict(k=cf.get("k", 10), mutual=cf.get("mutual", True), add_mst=cf.get("add_mst", True))
    if method == "snf":
        out["fused"] = snf_fuse(layers, t=cf.get("t", 10), alpha=cf.get("alpha", 0.5), **kw)
    elif method == "mean":
        out["fused"] = mean_fuse(layers, **kw)
    else:
        raise ValueError(f"неизвестный метод слияния: {method}")
    return out


# ----------------------------------------------------------------------------- метрики сети

def _laplacian_spectrum(A: sp.csr_matrix, n_eigs: int = 12) -> np.ndarray:
    """Наименьшие собственные значения нормированного лапласиана L = I − D^{-1/2} A D^{-1/2}.

    Считаются как 1 − (наибольшие с.з. нормированной смежности) через eigsh — быстрее и
    устойчивее, чем which="SM" для L. Для малых графов — плотный eigvalsh.
    """
    n = A.shape[0]
    d = np.asarray(A.sum(axis=1)).ravel()
    inv = 1.0 / np.sqrt(np.maximum(d, _EPS))
    An = sp.diags(inv) @ A @ sp.diags(inv)
    if n <= 300 or n_eigs >= n - 1:
        ev = np.linalg.eigvalsh(An.toarray())[::-1]
    else:
        ev = np.sort(eigsh(An, k=n_eigs, which="LA", return_eigenvectors=False))[::-1]
    lam = np.clip(1.0 - ev, 0.0, 2.0)
    return lam[:n_eigs]


def network_summary(A: sp.csr_matrix, X: np.ndarray | None = None) -> dict:
    """Сводка структуры графа: плотность, степени, компоненты, ассортативность, спектр, кластеризация.

    algebraic_connectivity — λ₂ нормированного лапласиана гигантской компоненты; spectral_gap —
    наибольший зазор λ_{k+1} − λ_k (k = 1..10) там же, spectral_gap_k — соответствующее k
    (эвристика числа кластеров, von Luxburg 2007). При заданном X добавляются средняя евклидова
    длина ребра и коэффициенты Рэлея xᵀLx / xᵀx (среднее по признакам; меньше — глаже на графе):
    rayleigh_quotient — комбинаторный L, rayleigh_quotient_norm — нормированный L (в [0, 2]).
    """
    import igraph as ig

    A = sp.csr_matrix(A)
    n = A.shape[0]
    U = sp.triu(A, k=1).tocoo()
    m = int(U.nnz)
    deg = np.asarray((A > 0).sum(axis=1)).ravel()
    out = {
        "n_nodes": int(n),
        "n_edges": m,
        "density": float(m / (n * (n - 1) / 2)) if n > 1 else 0.0,
        "mean_degree": float(deg.mean()) if n else 0.0,
        "mean_weight": float(U.data.mean()) if m else 0.0,
    }
    g = ig.Graph(n=n, edges=list(zip(U.row.tolist(), U.col.tolist())))
    g.es["weight"] = U.data.tolist()
    comps = g.connected_components()
    sizes = np.array(comps.sizes()) if n else np.array([0])
    out["n_components"] = int(len(sizes))
    out["giant_share"] = float(sizes.max() / n) if n else 0.0
    out["isolate_share"] = float((deg == 0).mean()) if n else 0.0
    out["degree_assortativity"] = float(g.assortativity_degree(directed=False)) if m else float("nan")
    out["avg_clustering"] = float(g.transitivity_avglocal_undirected(mode="zero")) if m else 0.0

    out["algebraic_connectivity"], out["spectral_gap"], out["spectral_gap_k"] = 0.0, 0.0, 0
    if m:
        giant = np.array(comps[int(sizes.argmax())])
        if len(giant) >= 3:
            lam = _laplacian_spectrum(A[giant][:, giant])
            out["algebraic_connectivity"] = float(lam[1])
            gaps = np.diff(lam)[: min(10, len(lam) - 1)]
            if len(gaps):
                kk = int(np.argmax(gaps))
                out["spectral_gap"], out["spectral_gap_k"] = float(gaps[kk]), kk + 1

    if X is not None and m:
        X = np.asarray(X, dtype=np.float64)
        out["mean_edge_length"] = float(np.linalg.norm(X[U.row] - X[U.col], axis=1).mean())
        d = np.asarray(A.sum(axis=1)).ravel()
        L = sp.diags(d) - A
        xx = np.maximum((X ** 2).sum(axis=0), _EPS)
        out["rayleigh_quotient"] = float(np.mean(np.einsum("if,if->f", X, L @ X) / xx))
        inv = 1.0 / np.sqrt(np.maximum(d, _EPS))
        Ln = sp.eye(n) - sp.diags(inv) @ A @ sp.diags(inv)
        out["rayleigh_quotient_norm"] = float(np.mean(np.einsum("if,if->f", X, Ln @ X) / xx))
    return out


def edge_jaccard(A: sp.csr_matrix, B: sp.csr_matrix) -> float:
    """Индекс Жаккара множеств рёбер (без учёта весов)."""
    a = sp.triu(sp.csr_matrix(A) > 0, k=1).astype(np.int8)
    b = sp.triu(sp.csr_matrix(B) > 0, k=1).astype(np.int8)
    inter = int(a.multiply(b).nnz)
    union = int(a.nnz + b.nnz - inter)
    return float(inter / union) if union else 1.0


def save_graph(A: sp.csr_matrix, path) -> None:
    """Сохранить граф в .npz (scipy.sparse.save_npz)."""
    sp.save_npz(str(path), sp.csr_matrix(A))


def load_graph(path) -> sp.csr_matrix:
    """Загрузить граф из .npz."""
    return sp.load_npz(str(path)).tocsr()
