"""Внутренние индексы качества кластеризации (ICVI) для атрибутированных сетей.

Признаковые: SW (silhouette), CH (Calinski–Harabasz), S_Dbw (Halkidi & Vazirgiannis, 2001).
Сетевые: AVI, AVU, ANUI (Biswas & Biswas, 2017; в нормировке библиотеки Pattern Лаборатории СберИндекс),
модулярность Q (Newman & Girvan, 2004), плотностная модулярность, MQ (Mancoridis et al., 1998/1999:
базовый MQ и TurboMQ).

Соглашения. A — симметричная матрица смежности с неотрицательными весами (numpy или scipy.sparse),
без петель; labels — целочисленные метки произвольных значений. Все формулы реализованы
самостоятельно по публикациям; с Pattern совпадают численно (см. tests/test_icvi.py).

Важные свойства (доказаны в тестах):
* AVU вырожден при K ≤ 3: AVU = 1 при K = 2 и AVU = 2/3 при K = 3 для любого разбиения с непустым разрезом.
* У случайного разбиения AVI ≈ 1/K, AVU ≈ (K−1)/(2K−3); поэтому индексы сравниваются при равном K и через
  z-оценки относительно перестановок меток (perm_null).
* TurboMQ на неориентированном графе тождественно равен K·AVI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import calinski_harabasz_score, silhouette_score

GRAPH_METRICS = ("AVI", "AVU", "ANUI", "Q", "Qdens", "MQ", "MQturbo")
FEATURE_METRICS = ("SW", "CH", "S_Dbw")
# направление: +1 — больше лучше, −1 — меньше лучше
DIRECTION = {"SW": 1, "CH": 1, "S_Dbw": -1, "AVI": 1, "AVU": -1, "ANUI": 1, "Q": 1, "Qdens": 1, "MQ": 1, "MQturbo": 1}


# ----------------------------------------------------------------------------- вспомогательное

def _encode(labels) -> tuple[np.ndarray, int]:
    _, enc = np.unique(np.asarray(labels), return_inverse=True)
    return enc.ravel(), int(enc.max()) + 1


def block_sums(A, labels) -> tuple[np.ndarray, np.ndarray]:
    """S[i, j] = суммарный вес рёбер между кластерами i и j (внутренние рёбра — дважды, как в Pattern).

    Возвращает (S, sizes). Работает с плотной и разреженной A.
    """
    enc, k = _encode(labels)
    n = len(enc)
    B = sp.csr_matrix((np.ones(n), (np.arange(n), enc)), shape=(n, k))
    A = sp.csr_matrix(A) if not sp.issparse(A) else A.tocsr()
    S = (B.T @ A @ B).toarray()
    return S, np.bincount(enc, minlength=k)


@dataclass
class GraphBlocks:
    S: np.ndarray      # K×K блочные суммы
    sizes: np.ndarray  # размеры кластеров
    @property
    def k(self) -> int: return len(self.sizes)
    @property
    def row(self) -> np.ndarray: return self.S.sum(axis=1)          # d_i: сумма степеней кластера
    @property
    def inner(self) -> np.ndarray: return np.diag(self.S)           # 2 × внутренний вес
    @property
    def out(self) -> np.ndarray: return self.row - self.inner        # вес разреза кластера
    @property
    def m2(self) -> float: return float(self.S.sum())                # 2m


def blocks(A, labels) -> GraphBlocks:
    S, sizes = block_sums(A, labels)
    return GraphBlocks(S, sizes)


# ----------------------------------------------------------------------------- сетевые индексы

def avi(A, labels) -> float:
    """Average Isolability: средняя по кластерам доля внутреннего веса в суммарном весе кластера."""
    b = blocks(A, labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        iso = np.where(b.row > 0, b.inner / b.row, 0.0)
    return float(iso.mean())


def avu(A, labels) -> float:
    """Average Unifiability (нормировка Pattern: сумма по упорядоченным парам, делённая на K).

    AVU = (1/K) Σ_i Σ_{j≠i} S_ij / (out_i + out_j − S_ij); слагаемое 0 при нулевом знаменателе.
    """
    b = blocks(A, labels)
    k = b.k
    if k < 2:
        return 0.0
    out = b.out
    den = out[:, None] + out[None, :] - b.S
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(den > 0, b.S / den, 0.0)
    np.fill_diagonal(term, 0.0)
    return float(term.sum() / k)


def avu_pairs(A, labels) -> float:
    """Вариант AVU с нормировкой на число упорядоченных пар K(K−1) — для сравнения между разными K."""
    b = blocks(A, labels)
    k = b.k
    return avu(A, labels) * k / (k * (k - 1)) if k > 1 else 0.0


def anui(A, labels) -> float:
    """ANUI = 1 / (AVU + 1/AVI); 0 при AVI = 0."""
    a = avi(A, labels)
    if a == 0:
        return 0.0
    return float(1.0 / (avu(A, labels) + 1.0 / a))


def modularity(A, labels) -> float:
    """Модулярность Ньюмана–Гирвана для взвешенного графа."""
    b = blocks(A, labels)
    if b.m2 == 0:
        return 0.0
    return float(np.sum(b.inner / b.m2 - (b.row / b.m2) ** 2))


def density_modularity(A, labels) -> float:
    """Плотностная модулярность (вариант Pattern): Σ_i (m_i − d_i²/4m) / n_i."""
    b = blocks(A, labels)
    if b.m2 == 0:
        return 0.0
    return float(np.sum((b.inner / 2 - b.row ** 2 / (2 * b.m2)) / b.sizes))


def mq_turbo(A, labels) -> float:
    """TurboMQ (Mancoridis et al., 1999): Σ_i CF_i, CF_i = μ_i / (μ_i + ½ Σ_{j≠i}(ε_ij + ε_ji)).

    Определение дано для орграфа. Неориентированное ребро трактуем как пару встречных дуг:
    μ_i = S_ii (внутренние дуги), ε_ij = ε_ji = S_ij, поэтому CF_i = S_ii / (S_ii + out_i) — это в точности
    изолируемость кластера, и TurboMQ ≡ K·AVI. Индекс оставлен для полноты; информации сверх AVI не несёт.
    """
    b = blocks(A, labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cf = np.where(b.row > 0, b.inner / b.row, 0.0)
    return float(cf.sum())


def mq_basic(A, labels) -> float:
    """Базовый MQ (Mancoridis et al., 1998): средняя внутрисвязность минус средняя межсвязность.

    A_i = μ_i / (n_i(n_i−1)/2) (плотность внутри кластера; 0 для одиночек),
    E_ij = ε_ij / (n_i n_j) (плотность между кластерами),
    MQ = (1/K) Σ_i A_i − (2/(K(K−1))) Σ_{i<j} E_ij.  Для взвешенных графов веса трактуются как кратности.
    """
    b = blocks(A, labels)
    k = b.k
    n = b.sizes.astype(float)
    mu = b.inner / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        intra = np.where(n > 1, mu / (n * (n - 1) / 2), 0.0)
    if k < 2:
        return float(intra.mean())
    inter = b.S / (n[:, None] * n[None, :])
    iu = np.triu_indices(k, 1)
    return float(intra.mean() - inter[iu].sum() / (k * (k - 1) / 2))


# ----------------------------------------------------------------------------- признаковые индексы

def s_dbw(X, labels, variant: str = "halkidi") -> float:
    """S_Dbw (Halkidi & Vazirgiannis, 2001) = Scat + Dens_bw. Меньше — лучше.

    variant="halkidi" — по статье: σ(·) — вектор дисперсий, density(·) считается по точкам c_i ∪ c_j
    для всех трёх опорных точек (v_i, v_j, u_ij).
    variant="package" — соглашения пакета `s-dbw` (method="Halkidi", nearest_centr=False): σ(·) — вектор СКО,
    density(v_i) считается только по точкам c_i. Нужен для сверки с чужими результатами.

    Scat = (1/K) Σ_i ||σ(c_i)|| / ||σ(X)||;  stdev = (1/K) sqrt(Σ_i ||σ(c_i)||);
    Dens_bw = 1/(K(K−1)) Σ_{i≠j} density(u_ij) / max(density(v_i), density(v_j)),
    density(u) = число точек в шаре радиуса stdev вокруг u; u_ij — середина отрезка между центрами.
    """
    X = np.asarray(X, dtype=float)
    enc, k = _encode(labels)
    if k < 2:
        return float("nan")
    spread = (lambda Z: Z.var(axis=0)) if variant == "halkidi" else (lambda Z: Z.std(axis=0))
    cents = np.vstack([X[enc == i].mean(axis=0) for i in range(k)])
    norm_i = np.array([np.linalg.norm(spread(X[enc == i])) for i in range(k)])
    scat = float(norm_i.mean() / np.linalg.norm(spread(X)))
    stdev = float(np.sqrt(norm_i.sum()) / k)

    def dens(points: np.ndarray, u: np.ndarray) -> int:
        return int((np.linalg.norm(points - u, axis=1) <= stdev).sum())

    own = [dens(X[enc == i], cents[i]) for i in range(k)]
    total = 0.0
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            pts = np.vstack([X[enc == i], X[enc == j]])
            u = (cents[i] + cents[j]) / 2
            if variant == "halkidi":
                dmax = max(dens(pts, cents[i]), dens(pts, cents[j]))
            else:
                dmax = max(own[i], own[j])
            total += dens(pts, u) / dmax if dmax > 0 else 0.0
    return scat + total / (k * (k - 1))


def feature_metrics(X, labels) -> dict[str, float]:
    enc, k = _encode(labels)
    if k < 2 or k >= len(enc):
        return {"SW": float("nan"), "CH": float("nan"), "S_Dbw": float("nan")}
    return {"SW": float(silhouette_score(X, enc)), "CH": float(calinski_harabasz_score(X, enc)), "S_Dbw": s_dbw(X, enc)}


def graph_metrics(A, labels) -> dict[str, float]:
    return {"AVI": avi(A, labels), "AVU": avu(A, labels), "ANUI": anui(A, labels), "Q": modularity(A, labels),
            "Qdens": density_modularity(A, labels), "MQ": mq_basic(A, labels), "MQturbo": mq_turbo(A, labels)}


def all_metrics(X, A, labels) -> dict[str, float]:
    return {**feature_metrics(X, labels), **graph_metrics(A, labels)}


# ----------------------------------------------------------------------------- нулевые модели

def perm_null(fn: Callable[[np.ndarray], float], labels, n_perm: int = 200, seed: int = 0) -> dict[str, float]:
    """Перестановочная нулевая модель: метки тасуются при сохранении размеров кластеров.

    Возвращает наблюдаемое значение, среднее и СКО по перестановкам, z-оценку и односторонние p-значения.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    obs = float(fn(labels))
    null = np.array([fn(rng.permutation(labels)) for _ in range(n_perm)], dtype=float)
    sd = null.std(ddof=1) if n_perm > 1 else float("nan")
    z = (obs - null.mean()) / sd if sd and sd > 0 else float("nan")
    p_hi = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    p_lo = float((np.sum(null <= obs) + 1) / (n_perm + 1))
    return {"obs": obs, "null_mean": float(null.mean()), "null_sd": float(sd), "z": float(z), "p_greater": p_hi, "p_less": p_lo}


def expected_random(k: int) -> dict[str, float]:
    """Аналитические ожидания для случайного разбиения на K равных кластеров (плотный однородный граф)."""
    return {"AVI": 1.0 / k, "AVU": (k - 1) / (2 * k - 3) if k >= 2 else 0.0, "MQturbo": 1.0}
