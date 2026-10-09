"""Выравнивание номеров кластеров между срезами (венгерское сопоставление по Жаккару).

Метки в разных срезах — произвольные целые; align_* перенумеровывают текущий срез так, чтобы
совпадающие кластеры носили номера опорного среза, а новые получали свежие номера.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _codes(labels) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    ids, inv = np.unique(labels, return_inverse=True)
    return ids, inv.ravel()


def contingency(a, b) -> np.ndarray:
    """Таблица сопряжённости K_a × K_b: число объектов в кластере i разбиения a и j разбиения b."""
    ids_a, ia = _codes(a)
    ids_b, ib = _codes(b)
    T = np.zeros((ids_a.size, ids_b.size), dtype=np.int64)
    np.add.at(T, (ia, ib), 1)
    return T


def jaccard_matrix(a, b) -> np.ndarray:
    """Матрица Жаккара K_a × K_b: |C_i ∩ C_j| / |C_i ∪ C_j| (кластеры упорядочены по возрастанию меток)."""
    T = contingency(a, b).astype(float)
    na = T.sum(axis=1, keepdims=True)
    nb = T.sum(axis=0, keepdims=True)
    return T / np.maximum(na + nb - T, 1.0)


def align_labels(prev, cur, criterion: str = "jaccard", next_label: int | None = None) -> np.ndarray:
    """Перенумеровать cur под prev: максимизируем суммарный Жаккар (или число совпадений, criterion="overlap")
    по сопоставлению кластеров (linear_sum_assignment). Кластеры cur без пары (или с нулевым пересечением)
    получают новые номера начиная с next_label (по умолчанию max(prev)+1), в порядке убывания размера."""
    prev = np.asarray(prev)
    cur = np.asarray(cur)
    if prev.shape != cur.shape:
        raise ValueError("prev и cur должны быть одной длины")
    ids_p, _ = _codes(prev)
    ids_c, ic = _codes(cur)
    S = jaccard_matrix(prev, cur) if criterion == "jaccard" else contingency(prev, cur).astype(float)
    rows, cols = linear_sum_assignment(-S)
    mapping = np.full(ids_c.size, -1, dtype=np.int64)
    for r, c in zip(rows, cols):
        if S[r, c] > 0:
            mapping[c] = ids_p[r]
    nxt = int(ids_p.max()) + 1 if next_label is None else int(next_label)
    sizes = np.bincount(ic, minlength=ids_c.size)
    for c in sorted(np.flatnonzero(mapping < 0), key=lambda j: (-sizes[j], j)):
        mapping[c] = nxt
        nxt += 1
    return mapping[ic]


def align_to_reference(ref, cur, criterion: str = "jaccard") -> np.ndarray:
    """Выровнять cur под фиксированный опорный срез ref (например, первый месяц или консенсус)."""
    return align_labels(ref, cur, criterion=criterion)


def align_sequence(labels_list: list, criterion: str = "jaccard") -> list[np.ndarray]:
    """Последовательное выравнивание: каждый срез под уже выровненный предыдущий.
    Новые кластеры получают номера, не использованные ни в одном из предыдущих срезов."""
    if not labels_list:
        return []
    out = [np.asarray(labels_list[0]).astype(np.int64)]
    used_max = int(out[0].max())
    for cur in labels_list[1:]:
        aligned = align_labels(out[-1], cur, criterion=criterion, next_label=used_max + 1)
        used_max = max(used_max, int(aligned.max()))
        out.append(aligned)
    return out
