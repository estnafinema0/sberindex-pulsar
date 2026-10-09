"""Методы кластеризации атрибутированной сети МО: единый интерфейс run_method / run_all.

Соглашения. X — (N, F) стандартизованные признаки; A — scipy.sparse.csr_matrix (N, N), симметричная
взвешенная смежность без диагонали; k — требуемое число кластеров; seed — зерно генератора.
Все методы возвращают целочисленные метки 0..k'−1 (k' = k, кроме Leiden, где k' — достигнутое число).

Методы:
* kmeans, ward      — только признаки X (базовые);
* spectral          — только сеть A (sklearn, precomputed affinity, cluster_qr);
* leiden            — только сеть A; разрешение γ подбирается бисекцией под k крупных сообществ
                      (RBConfigurationVertexPartition, Traag et al. 2019), мелкие присоединяются;
* kefrin            — признаки + сеть: собственная реализация KEFRiN (Shalileh & Mirkin, Entropy 2022,
                      ур. (5)–(7)): F = ρ‖Y − SC‖² + ξ‖P − SΛ‖², P — модулярностная матрица сети;
* run_temporal_leiden — мультисрезовая модулярность Mucha et al. 2010 (Leiden по всем срезам сразу,
                      межсрезовый вес ω), метки сквозные по построению.

Код написан по формулам из публикаций; чужие реализации не использовались.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable

import igraph as ig
import leidenalg as la
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering

METHODS = ["kmeans", "ward", "spectral", "leiden", "kefrin", "kefrin_balanced"]


# ----------------------------------------------------------------------------- вспомогательное

def _as_csr(A) -> sp.csr_matrix:
    A = sp.csr_matrix(A, dtype=np.float64)
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


def _relabel(labels) -> np.ndarray:
    """Перенумеровать метки в 0..k'−1 (по возрастанию исходных значений)."""
    _, enc = np.unique(np.asarray(labels), return_inverse=True)
    return enc.ravel().astype(np.int64)


def _indicator(labels: np.ndarray, k: int) -> sp.csr_matrix:
    n = labels.shape[0]
    return sp.csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, k))


def _sqdist(Z: np.ndarray, z2: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Квадраты евклидовых расстояний строк Z до строк C: ‖z‖² + ‖c‖² − 2 z·c (z2 = ‖z_i‖² заранее)."""
    c2 = np.einsum("kd,kd->k", C, C)
    D = z2[:, None] + c2[None, :] - 2.0 * (Z @ C.T)
    np.maximum(D, 0.0, out=D)
    return D


# ----------------------------------------------------------------------------- базовые методы

def run_kmeans(X, k: int, seed: int = 0, n_init: int = 10) -> np.ndarray:
    """K-means на признаках (sklearn, k-means++, n_init запусков, лучший по инерции)."""
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(np.asarray(X, dtype=float))


def run_ward(X, k: int) -> np.ndarray:
    """Иерархическая агломерация Уорда на признаках (детерминирована, seed не нужен)."""
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(np.asarray(X, dtype=float))


def run_spectral(A, k: int, seed: int = 0, eps: float = 1e-6) -> np.ndarray:
    """Спектральная кластеризация на готовой матрице сходства (нормированный лапласиан, cluster_qr).

    Если граф несвязный, ко всем парам добавляется малая ε-связность (eps), чтобы спектральное
    вложение было определено корректно и sklearn не предупреждал о несвязности; на связном графе
    матрица передаётся как есть.
    """
    A = _as_csr(A)
    n_comp, _ = connected_components(A, directed=False)
    dense = A.toarray()
    if n_comp > 1:
        warnings.warn(f"spectral: граф несвязный ({n_comp} компонент) — добавлена ε-связность {eps:g}; "
                      "мелкие компоненты могут стать отдельными кластерами", stacklevel=2)
        dense = dense + eps
        np.fill_diagonal(dense, 0.0)
    model = SpectralClustering(n_clusters=k, affinity="precomputed", assign_labels="cluster_qr", random_state=seed)
    return model.fit_predict(dense)


# ----------------------------------------------------------------------------- Leiden с подбором γ

def _to_igraph(A) -> ig.Graph:
    """Неориентированный взвешенный igraph-граф из симметричной csr (берётся верхний треугольник)."""
    A = _as_csr(A)
    U = sp.triu(A, k=1).tocoo()
    g = ig.Graph(n=A.shape[0], edges=list(zip(U.row.tolist(), U.col.tolist())))
    g.es["weight"] = U.data.astype(float).tolist()
    return g


def _leiden(g: ig.Graph, gamma: float, seed: int, n_iterations: int) -> np.ndarray:
    part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                             resolution_parameter=float(gamma), seed=int(seed), n_iterations=n_iterations)
    return np.asarray(part.membership, dtype=np.int64)


def merge_small_clusters(A, labels, min_size: int = 5, relabel: bool = True) -> np.ndarray:
    """Присоединить кластеры размером < min_size к крупному кластеру с максимальным суммарным весом рёбер.

    Если кластер не связан ни с одним крупным — к самому большому. Если крупных нет — метки не меняются.
    """
    A = _as_csr(A)
    labels = np.asarray(labels, dtype=np.int64)
    ids, inv, sizes = np.unique(labels, return_inverse=True, return_counts=True)
    large = np.flatnonzero(sizes >= min_size)
    small = np.flatnonzero(sizes < min_size)
    if large.size == 0 or small.size == 0:
        return _relabel(labels) if relabel else labels
    M = _indicator(inv, ids.size)
    B = np.asarray((M.T @ A @ M).todense())            # суммарные веса между кластерами
    new_inv = inv.copy()
    for s in small:
        w = B[s, large]
        target = large[int(np.argmax(w))] if w.max() > 0 else large[int(np.argmax(sizes[large]))]
        new_inv[inv == s] = target
    out = ids[new_inv]
    return _relabel(out) if relabel else out


def _bisect_gamma(evaluate: Callable[[float], tuple[int, bool, object]], k: int, gamma_lo: float,
                  gamma_hi: float, max_iter: int) -> tuple[float, object, int]:
    """Бисекция по log γ: ищем γ с evaluate(γ)[0] == k (число крупных сообществ).

    evaluate(γ) -> (n_large, over, payload); over=True означает «граф уже раздроблен» (средний размер
    сообщества < min_size), тогда γ надо уменьшать даже при n_large < k. Число вызовов ≤ max_iter;
    при недостижимости точного k возвращается γ с ближайшим n_large (при равенстве — меньшая γ).
    """
    best: tuple[float, object, int] | None = None
    calls = 0

    def probe(g: float):
        nonlocal best, calls
        calls += 1
        n_large, over, payload = evaluate(g)
        if best is None or abs(n_large - k) < abs(best[2] - k):
            best = (g, payload, n_large)
        return n_large, over

    lo, hi = float(gamma_lo), float(gamma_hi)
    c_lo, over_lo = probe(lo)
    if c_lo == k:
        return best
    while (c_lo > k or over_lo) and calls < max_iter:     # нижняя граница слишком велика
        lo /= 10.0
        c_lo, over_lo = probe(lo)
        if c_lo == k:
            return best
    c_hi, over_hi = probe(hi)
    if c_hi == k:
        return best
    while c_hi < k and not over_hi and calls < max_iter:  # верхняя граница слишком мала
        hi *= 10.0
        c_hi, over_hi = probe(hi)
        if c_hi == k:
            return best
    while calls < max_iter and hi / lo > 1.0 + 1e-6:
        mid = math.sqrt(lo * hi)
        c, over = probe(mid)
        if c == k:
            return best
        if c < k and not over:
            lo = mid
        else:
            hi = mid
    return best


def _leiden_search(A, k: int, min_size: int, seed: int, gamma_lo: float, gamma_hi: float,
                   max_iter: int, n_iterations: int) -> tuple[float, np.ndarray, int]:
    A = _as_csr(A)
    g = _to_igraph(A)
    n = A.shape[0]

    def evaluate(gamma: float):
        memb = _leiden(g, gamma, seed, n_iterations)
        sizes = np.bincount(memb)
        n_large = int((sizes >= min_size).sum())
        over = sizes.size * min_size > n
        return n_large, over, memb

    gamma, memb, n_large = _bisect_gamma(evaluate, k, gamma_lo, gamma_hi, max_iter)
    return gamma, memb, n_large


def leiden_resolution_for_k(A, k: int, min_size: int = 5, seed: int = 0, gamma_lo: float = 1e-2,
                            gamma_hi: float = 10.0, max_iter: int = 25, n_iterations: int = -1) -> float:
    """Разрешение γ (RB-модулярность), при котором Leiden даёт ровно k сообществ размером ≥ min_size.

    Бисекция по log γ, ≤ max_iter запусков Leiden с фиксированным seed. Если точное k не достигается
    (счётчик не строго монотонен по γ), возвращается ближайшее.
    """
    return _leiden_search(A, k, min_size, seed, gamma_lo, gamma_hi, max_iter, n_iterations)[0]


def run_leiden(A, k: int, seed: int = 0, min_size: int = 5, gamma_lo: float = 1e-2, gamma_hi: float = 10.0,
               max_iter: int = 25, n_iterations: int = -1, params: dict | None = None) -> np.ndarray:
    """Leiden на сети с подбором γ под k; мелкие сообщества присоединяются к ближайшему крупному.

    Возвращает метки 0..k'−1, k' — достигнутое число крупных сообществ. Если передан params (dict),
    в него записываются params["gamma_"] и params["k_achieved_"].
    """
    gamma, memb, n_large = _leiden_search(A, k, min_size, seed, gamma_lo, gamma_hi, max_iter, n_iterations)
    labels = merge_small_clusters(A, memb, min_size=min_size)
    if params is not None:
        params["gamma_"] = float(gamma)
        params["k_achieved_"] = int(labels.max()) + 1
    return labels


# ----------------------------------------------------------------------------- temporal Leiden (Mucha 2010)

def run_temporal_leiden(A_list: list, k: int, interslice_weight: float = 0.5, seed: int = 0, min_size: int = 5,
                        gamma_lo: float = 1e-2, gamma_hi: float = 10.0, max_iter: int = 25,
                        n_iterations: int = -1, params: dict | None = None) -> list[np.ndarray]:
    """Мультисрезовая модулярность Mucha et al. (2010): Leiden по всем срезам сразу.

    Q = (1/2μ) Σ_{ijsr} [(A_ijs − γ k_is k_js / 2m_s) δ_sr + δ_ij ω δ_{s,r±1}] δ(g_is, g_jr):
    внутри среза — RB-модулярность с разрешением γ, между соседними срезами узел связан сам с собой
    весом ω (interslice_weight) без нуль-модели. Узлы сопоставляются по индексу (атрибут id).
    γ подбирается бисекцией так, чтобы медианное по срезам число сообществ размером ≥ min_size
    равнялось k. Мелкие сообщества в каждом срезе присоединяются к крупному по весу рёбер этого среза.
    Метки сквозные: один и тот же номер в разных срезах — одно и то же сообщество.
    """
    mats = [_as_csr(A) for A in A_list]
    n = mats[0].shape[0]
    if any(A.shape[0] != n for A in mats):
        raise ValueError("все срезы должны иметь одинаковое число узлов")
    graphs = []
    for A in mats:
        g = _to_igraph(A)
        g.vs["id"] = list(range(n))
        graphs.append(g)

    def evaluate(gamma: float):
        membs, _ = la.find_partition_temporal(graphs, la.RBConfigurationVertexPartition,
                                              interslice_weight=float(interslice_weight), vertex_id_attr="id",
                                              weight_attr="weight", n_iterations=n_iterations, seed=int(seed),
                                              resolution_parameter=float(gamma))
        membs = [np.asarray(m, dtype=np.int64) for m in membs]
        n_large = [int((np.bincount(m) >= min_size).sum()) for m in membs]
        n_total = [int(np.unique(m).size) for m in membs]
        med = float(np.median(n_large))
        over = float(np.median(n_total)) * min_size > n
        # медиана может быть дробной — тогда она заведомо != k; направление определяется сравнением
        return (k if med == k else int(math.floor(med)) if med < k else int(math.ceil(med))), over, membs

    gamma, membs, _ = _bisect_gamma(evaluate, k, gamma_lo, gamma_hi, max_iter)
    merged = [merge_small_clusters(A, m, min_size=min_size, relabel=False) for A, m in zip(mats, membs)]
    ids = np.unique(np.concatenate(merged))
    out = [np.searchsorted(ids, m).astype(np.int64) for m in merged]
    if params is not None:
        params["gamma_"] = float(gamma)
        params["k_achieved_"] = int(ids.size)
    return out


# ----------------------------------------------------------------------------- KEFRiN

def modularity_matrix(A) -> np.ndarray:
    """Модулярностная трансформация сети: P = A − d dᵀ / (2m) (плотная N×N; d — взвешенные степени)."""
    A = _as_csr(A)
    d = np.asarray(A.sum(axis=1)).ravel()
    two_m = d.sum()
    P = A.toarray()
    if two_m > 0:
        P -= np.outer(d, d) / two_m
    return P


def _normalize_rows(Z: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(nrm, 1e-12)


@dataclass
class KefrinResult:
    labels: np.ndarray                     # метки 0..k−1
    objective: float                       # значение критерия F
    history: list[float] = field(default_factory=list)   # F после каждого шага назначения (лучший запуск)
    n_iter: int = 0
    centers_X: np.ndarray | None = None    # C (K×F)
    centers_P: np.ndarray | None = None    # Λ (K×N)


def _kefrin_prepare(X, A, rho: float, xi: float, metric: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    Y = np.asarray(X, dtype=np.float64) if rho > 0 else None
    P = modularity_matrix(A) if xi > 0 else None
    if metric == "cosine":
        Y = _normalize_rows(Y) if Y is not None else None
        P = _normalize_rows(P) if P is not None else None
    elif metric != "euclidean":
        raise ValueError(f"metric должен быть 'euclidean' или 'cosine', получено {metric!r}")
    return Y, P


def _kefrin_init(Y, P, y2, p2, k: int, rho: float, xi: float, rng: np.random.Generator, init: str) -> np.ndarray:
    """Индексы начальных центров: K-Means++ (вероятностно, ∝ объединённому расстоянию) или MaxMin."""
    n = (Y if Y is not None else P).shape[0]
    chosen = [int(rng.integers(n))]
    dmin = np.full(n, np.inf)
    dsum = np.zeros(n)
    while len(chosen) < k:
        j = chosen[-1]
        d = np.zeros(n)
        if Y is not None:
            d += rho * np.maximum(y2 + y2[j] - 2.0 * (Y @ Y[j]), 0.0)
        if P is not None:
            d += xi * np.maximum(p2 + p2[j] - 2.0 * (P @ P[j]), 0.0)
        dmin = np.minimum(dmin, d)
        dsum += d
        if init == "maxmin":
            cand = dsum.copy()
            cand[chosen] = -np.inf
            nxt = int(np.argmax(cand))
        else:
            w = dmin.copy()
            w[chosen] = 0.0
            tot = w.sum()
            if tot <= 0:
                w = np.ones(n)
                w[chosen] = 0.0
                tot = w.sum()
            nxt = int(rng.choice(n, p=w / tot))
        chosen.append(nxt)
    return np.asarray(chosen)


def _kefrin_once(Y, P, k: int, rho: float, xi: float, rng: np.random.Generator, max_iter: int, tol: float,
                 metric: str, init: str) -> KefrinResult:
    """Один запуск чередующейся минимизации F при фиксированной инициализации."""
    n = (Y if Y is not None else P).shape[0]
    y2 = np.einsum("ij,ij->i", Y, Y) if Y is not None else None
    p2 = np.einsum("ij,ij->i", P, P) if P is not None else None
    idx = _kefrin_init(Y, P, y2, p2, k, rho, xi, rng, init)
    C = Y[idx].copy() if Y is not None else None
    L = P[idx].copy() if P is not None else None
    labels = np.full(n, -1, dtype=np.int64)
    history: list[float] = []
    it = 0
    for it in range(1, max_iter + 1):
        # (а) назначение: i → argmin_k ρ d(y_i, c_k) + ξ d(p_i, λ_k)
        D = np.zeros((n, k))
        if Y is not None:
            D += rho * _sqdist(Y, y2, C)
        if P is not None:
            D += xi * _sqdist(P, p2, L)
        new = np.argmin(D, axis=1)
        dmin = D[np.arange(n), new]
        # пустые кластеры: отдаём им самые дальние объекты
        counts = np.bincount(new, minlength=k)
        for empty in np.flatnonzero(counts == 0):
            far = int(np.argmax(dmin))
            new[far] = empty
            dmin[far] = 0.0
        F = float(dmin.sum())
        history.append(F)
        converged = np.array_equal(new, labels) or (len(history) > 1 and history[-2] - F <= tol * max(F, 1.0))
        labels = new
        # (б) центры — внутрикластерные средние (для cosine — с повторной нормировкой)
        M = _indicator(labels, k)
        inv_n = 1.0 / np.maximum(np.bincount(labels, minlength=k), 1)
        if Y is not None:
            C = np.asarray(M.T @ Y) * inv_n[:, None]
            if metric == "cosine":
                C = _normalize_rows(C)
        if P is not None:
            L = np.asarray(M.T @ P) * inv_n[:, None]
            if metric == "cosine":
                L = _normalize_rows(L)
        if converged:
            break
    return KefrinResult(labels=labels, objective=history[-1], history=history, n_iter=it, centers_X=C, centers_P=L)


def kefrin(X, A, k: int, rho: float = 1.0, xi: float = 1.0, seed: int = 0, n_init: int = 5, max_iter: int = 100,
           tol: float = 1e-10, metric: str = "euclidean", init: str = "kmeans++") -> KefrinResult:
    """KEFRiN (Shalileh & Mirkin, 2022): K-means в объединённом пространстве «признаки + строки P».

    Критерий F = ρ‖Y − SC‖² + ξ‖P − SΛ‖² (ур. (5)), P = A − ddᵀ/2m (модулярностная трансформация);
    чередование правила минимального расстояния (ур. (6)) и пересчёта центров как средних (ур. (7))
    до неизменности меток или max_iter; из n_init запусков (K-Means++ / MaxMin) берётся минимум F.
    metric="cosine": строки Y, P и центры нормируются (d_c = 1 − cos = ½‖ŷ − ĉ‖²), центры после
    усреднения нормируются повторно — как в KEFRiNc; монотонность F при этом не гарантирована.
    ρ=0 или ξ=0 отключают соответствующий источник (матрица P тогда не строится).
    """
    if rho < 0 or xi < 0 or rho + xi == 0:
        raise ValueError("нужны ρ, ξ ≥ 0, не оба нулевые")
    Y, P = _kefrin_prepare(X, A, rho, xi, metric)
    best: KefrinResult | None = None
    for ss in np.random.SeedSequence(seed).spawn(n_init):
        res = _kefrin_once(Y, P, k, rho, xi, np.random.default_rng(ss), max_iter, tol, metric, init)
        if best is None or res.objective < best.objective:
            best = res
    best.labels = _relabel(best.labels) if np.unique(best.labels).size == k else best.labels
    return best


def kefrin_objective(X, A, labels, rho: float = 1.0, xi: float = 1.0, metric: str = "euclidean") -> float:
    """Значение критерия F = ρ‖Y − SC‖² + ξ‖P − SΛ‖² при центрах-средних для заданных меток."""
    labels = _relabel(labels)
    k = int(labels.max()) + 1
    Y, P = _kefrin_prepare(X, A, rho, xi, metric)
    M = _indicator(labels, k)
    inv_n = 1.0 / np.maximum(np.bincount(labels, minlength=k), 1)
    F = 0.0
    for Z, w in ((Y, rho), (P, xi)):
        if Z is None or w == 0:
            continue
        Cz = np.asarray(M.T @ Z) * inv_n[:, None]
        if metric == "cosine":
            Cz = _normalize_rows(Cz)
        R = Z - Cz[labels]
        F += w * float(np.einsum("ij,ij->", R, R))
    return F


# ----------------------------------------------------------------------------- диспетчер

def run_method(name: str, X, A, k: int, seed: int = 0, params: dict | None = None) -> np.ndarray:
    """Запустить метод по имени из METHODS; params — словарь гиперпараметров метода (см. configs/methods.yaml).

    Для leiden в params записывается достигнутая γ (ключ gamma_), для kefrin — значение F (objective_).
    """
    p = dict(params or {})
    p = {key: val for key, val in p.items() if not key.endswith("_")}   # служебные выходы не передаём
    if name == "kmeans":
        return run_kmeans(X, k, seed=seed, **p)
    if name == "ward":
        return run_ward(X, k, **p)
    if name == "spectral":
        return run_spectral(A, k, seed=seed, **p)
    if name == "leiden":
        return run_leiden(A, k, seed=seed, params=params, **p)
    if name in ("kefrin", "kefrin_balanced"):
        if name == "kefrin_balanced" or p.pop("balance", False):
            # «сбалансированный» вариант: ξ масштабируется так, чтобы слагаемые ρ‖Y‖² и ξ‖P‖² были равны;
            # в статье ρ = ξ = 1, но для разреженной сети ‖P‖² ≪ ‖Y‖², и сеть почти не влияет на решение
            P = modularity_matrix(A)
            scale = float((np.asarray(X) ** 2).sum() / max((P ** 2).sum(), 1e-12))
            p["xi"] = float(p.get("xi", 1.0)) * scale
            if params is not None:
                params["xi_effective_"] = p["xi"]
        res = kefrin(X, A, k, seed=seed, **p)
        if params is not None:
            params["objective_"] = res.objective
        return res.labels
    raise ValueError(f"неизвестный метод {name!r}; доступны {METHODS}")


def run_all(X, A, k: int, seed: int = 0, cfg: dict | None = None) -> dict[str, np.ndarray]:
    """Все методы из cfg['methods'] (по умолчанию METHODS) с параметрами cfg['params'][name] → {name: метки}."""
    cfg = cfg or {}
    names = cfg.get("methods", METHODS)
    per_method = cfg.get("params", {}) or {}
    return {name: run_method(name, X, A, k, seed=seed, params=dict(per_method.get(name, {}) or {})) for name in names}
