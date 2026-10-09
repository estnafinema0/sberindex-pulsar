"""Оценка качества разбиений: панель ICVI, нулевые модели, бутстрэп-ДИ, сводные рейтинги.

Используется на этапе сравнения методов. Все функции чистые (принимают массивы), чтобы их можно было тестировать.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import davies_bouldin_score

from pulsar import icvi

# Панель индексов, которую ждёт жюри, + дополнения из Doklady Math 2025 / ESWA 2026.
PANEL = ["SW", "CH", "S_Dbw", "AVI", "AVU", "MQ", "ANUI", "DBI", "CH_N", "Qdens"]
DIRECTION = {**icvi.DIRECTION, "DBI": -1, "CH_N": 1, "MQ": 1}


def metric_panel(X: np.ndarray, A, labels: np.ndarray) -> dict[str, float]:
    """Все индексы для одного разбиения. MQ = модулярность Q (трактовка работ Шалилеха);
    TurboMQ и базовый MQ Манкоридиса — отдельными колонками для приложения."""
    enc, k = icvi._encode(labels)
    fm = icvi.feature_metrics(X, enc)
    gm = icvi.graph_metrics(A, enc)
    out = {**fm, "AVI": gm["AVI"], "AVU": gm["AVU"], "ANUI": gm["ANUI"], "MQ": gm["Q"], "Qdens": gm["Qdens"],
           "MQ_turbo": gm["MQturbo"], "MQ_mancoridis": gm["MQ"]}
    out["DBI"] = float(davies_bouldin_score(X, enc)) if 1 < k < len(enc) else float("nan")
    out["CH_N"] = out["CH"] / len(enc)
    out["K"] = k
    return out


# ----------------------------------------------------------------------------- нулевые модели

def permutation_null(X, A, labels, n_perm: int = 100, seed: int = 0, metrics: Iterable[str] = PANEL) -> pd.DataFrame:
    """Перестановка меток при сохранении размеров кластеров → среднее, СКО, z и p для каждого индекса."""
    rng = np.random.default_rng(seed)
    obs = metric_panel(X, A, labels)
    null = []
    for _ in range(n_perm):
        null.append(metric_panel(X, A, rng.permutation(labels)))
    null = pd.DataFrame(null)
    rows = []
    for m in metrics:
        mu, sd = null[m].mean(), null[m].std(ddof=1)
        z = (obs[m] - mu) / sd if sd > 0 else np.nan
        better = (null[m] >= obs[m]) if DIRECTION[m] > 0 else (null[m] <= obs[m])
        rows.append({"metric": m, "obs": obs[m], "null_mean": mu, "null_sd": sd, "z": z * DIRECTION[m],
                     "p": (better.sum() + 1) / (n_perm + 1)})
    return pd.DataFrame(rows)


def configuration_model_null(A, labels, n_rep: int = 50, seed: int = 0) -> pd.DataFrame:
    """Нулевая модель графа: перестановка рёбер с сохранением степеней (double edge swap, igraph rewire),
    метки фиксированы → насколько сетевые индексы объясняются только распределением степеней."""
    import igraph as ig
    rng = np.random.default_rng(seed)
    A = sp.csr_matrix(A)
    src, dst = sp.triu(A, 1).nonzero()
    w = np.asarray(A[src, dst]).ravel()
    g = ig.Graph(n=A.shape[0], edges=list(zip(src.tolist(), dst.tolist())))
    g.es["weight"] = w.tolist()
    obs = icvi.graph_metrics(A, labels)
    rows = []
    for r in range(n_rep):
        h = g.copy()
        h.rewire(n=10 * h.ecount(), allowed_edge_types="simple")  # сохраняет степени
        ws = rng.permutation(w)                      # веса перемешиваем независимо
        e = np.array(h.get_edgelist())
        B = sp.coo_matrix((ws, (e[:, 0], e[:, 1])), shape=A.shape)
        B = (B + B.T).tocsr()
        rows.append(icvi.graph_metrics(B, labels))
    null = pd.DataFrame(rows)
    out = []
    for m in ("AVI", "AVU", "ANUI", "Q", "Qdens"):
        mu, sd = null[m].mean(), null[m].std(ddof=1)
        out.append({"metric": m, "obs": obs[m], "null_mean": mu, "null_sd": sd, "z": ((obs[m] - mu) / sd if sd > 0 else np.nan) * icvi.DIRECTION[m]})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- бутстрэп

def paired_bootstrap(values: pd.DataFrame, n_boot: int = 2000, seed: int = 0, ci: float = 0.95) -> pd.DataFrame:
    """Парный бутстрэп по строкам (месяцам или подвыборкам): ДИ разности «метод A − метод B» для каждой пары колонок.

    values: индекс — месяц (или повтор), колонки — методы, значения — индекс качества (уже с учётом направления).
    """
    rng = np.random.default_rng(seed)
    cols = list(values.columns)
    V = values.to_numpy(float)
    n = V.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = V[idx].mean(axis=1)  # n_boot × methods
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    rows = []
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            d = boot_means[:, i] - boot_means[:, j]
            rows.append({"A": a, "B": b, "diff": float(V[:, i].mean() - V[:, j].mean()),
                         "ci_lo": float(np.quantile(d, lo)), "ci_hi": float(np.quantile(d, hi)),
                         "p_A_better": float((d > 0).mean()), "significant": bool(np.quantile(d, lo) > 0 or np.quantile(d, hi) < 0)})
    return pd.DataFrame(rows)


def bootstrap_win_rate(values: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.Series:
    """Как часто метод оказывается лучшим по среднему на бутстрэп-подвыборке строк."""
    rng = np.random.default_rng(seed)
    V = values.to_numpy(float); n = V.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    winners = V[idx].mean(axis=1).argmax(axis=1)
    return pd.Series(np.bincount(winners, minlength=V.shape[1]) / n_boot, index=values.columns)


# ----------------------------------------------------------------------------- сводные рейтинги

def oriented(table: pd.DataFrame, metrics: Iterable[str] = PANEL) -> pd.DataFrame:
    """Привести индексы к виду «больше — лучше» (умножить на направление)."""
    t = table.copy()
    for m in metrics:
        if m in t:
            t[m] = t[m] * DIRECTION[m]
    return t


def borda(table: pd.DataFrame, metrics: Iterable[str] = PANEL) -> pd.Series:
    """Правило Борды: по каждому индексу методы ранжируются (1 — лучший), очки = K_методов − ранг; сумма по индексам.
    table: индекс — метод, колонки — индексы (сырые, направление учитывается здесь)."""
    t = oriented(table, metrics)
    n = len(t)
    score = pd.Series(0.0, index=t.index)
    for m in metrics:
        if m in t:
            score += n - t[m].rank(ascending=False, method="average")
    return score.sort_values(ascending=False)


def threshold_aggregation(table: pd.DataFrame, metrics: Iterable[str] = PANEL, grades: int = 3) -> pd.DataFrame:
    """Пороговое правило Алескерова (некомпенсаторное): каждому методу по каждому индексу ставится оценка 1..grades
    (терцили среди методов, 1 — худшая); методы упорядочиваются лексикографически: сначала меньше оценок «1»,
    при равенстве — меньше оценок «2», и т.д. Возвращает таблицу с числом оценок каждого уровня и итоговым рангом."""
    t = oriented(table, metrics)
    ms = [m for m in metrics if m in t]
    G = pd.DataFrame(index=t.index)
    for m in ms:
        G[m] = pd.qcut(t[m].rank(method="first"), grades, labels=range(1, grades + 1)).astype(int)
    counts = pd.DataFrame({f"n{g}": (G == g).sum(axis=1) for g in range(1, grades + 1)})
    order = counts.sort_values([f"n{g}" for g in range(1, grades)], ascending=True)
    order["rank"] = np.arange(1, len(order) + 1)
    return order


def summarize_over_months(long: pd.DataFrame, by: tuple[str, ...] = ("method", "K"), metrics: Iterable[str] = PANEL) -> pd.DataFrame:
    """Медиана и межквартильный размах по месяцам."""
    g = long.groupby(list(by))
    med = g[list(metrics)].median()
    iqr = g[list(metrics)].quantile(0.75) - g[list(metrics)].quantile(0.25)
    iqr.columns = [f"{c}_iqr" for c in iqr.columns]
    return med.join(iqr)
