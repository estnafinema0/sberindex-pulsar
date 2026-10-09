"""Динамика типов МО во времени: переходы, подтверждённые смены, события Greene, бутстрэп, out-of-time.

Чистые функции над массивами. Соглашения:
  L — (T, N) целочисленные сквозные метки (после align_sequence или из temporal Leiden);
  X — (T, N, F) признаки; months — список 'YYYY-MM' длины T.
Формулы: Shorrocks (1978) M = (K − tr P)/(K − 1); Greene et al. (2010) — сопоставление сообществ
соседних срезов по Жаккару J ≥ θ; правило подтверждения — новый тип держится ≥ min_run месяцев подряд.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from pulsar import align

NEW_YEAR = "12→01"


# ----------------------------------------------------------------------------- служебное

def _as_TN(L) -> np.ndarray:
    L = np.asarray(L)
    if L.ndim == 1:
        L = L[None, :]
    if L.ndim != 2:
        raise ValueError("L должна быть (T, N)")
    return L.astype(np.int64)


def _month_labels(months, T: int) -> list:
    if months is None:
        return [str(t) for t in range(T)]
    if len(months) != T:
        raise ValueError("len(months) != T")
    return [str(m) for m in months]


def _cal(month: str) -> int:
    """Календарный месяц из 'YYYY-MM'."""
    return int(str(month)[5:7])


def _runs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Отрезки постоянства ряда: (начала, длины, значения)."""
    T = x.size
    ch = np.flatnonzero(x[1:] != x[:-1]) + 1
    starts = np.r_[0, ch]
    ends = np.r_[ch, T]
    return starts, ends - starts, x[starts]


def _remaining_run(L: np.ndarray) -> np.ndarray:
    """rem[t, i] — сколько месяцев подряд, начиная с t, МО остаётся в типе L[t, i] (включая t)."""
    T = L.shape[0]
    rem = np.ones_like(L)
    for t in range(T - 2, -1, -1):
        rem[t] = np.where(L[t] == L[t + 1], rem[t + 1] + 1, 1)
    return rem


def month_indices(months, start: str, end: str) -> np.ndarray:
    """Индексы месяцев в [start, end] включительно (строки 'YYYY-MM' сравниваются лексикографически)."""
    m = np.asarray([str(x) for x in months])
    return np.flatnonzero((m >= start) & (m <= end))


# ----------------------------------------------------------------------------- 1. переходы и мобильность

def transition_counts(a, b, ids=None) -> pd.DataFrame:
    """Таблица переходов между двумя срезами меток a → b (строки — откуда)."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    ids = np.unique(np.r_[a, b]) if ids is None else np.asarray(ids)
    ia, ib = np.searchsorted(ids, a), np.searchsorted(ids, b)
    C = np.zeros((ids.size, ids.size), dtype=np.int64)
    np.add.at(C, (ia, ib), 1)
    return pd.DataFrame(C, index=ids, columns=ids)


def _normalize_rows(C: pd.DataFrame) -> pd.DataFrame:
    s = C.sum(axis=1).to_numpy(dtype=float)
    P = C.to_numpy(dtype=float) / np.where(s > 0, s, np.nan)[:, None]   # пустые строки → NaN
    return pd.DataFrame(P, index=C.index, columns=C.columns)


def transition_matrix(L, lag: int = 1, normalize: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Матрица переходов через lag месяцев, пул по всем парам (t, t+lag).

    Возвращает (P, counts): K×K по объединению меток; P — доли по строкам (откуда), строки без
    наблюдений = NaN (типы, появившиеся только в конце); при normalize=False P = counts.
    """
    L = _as_TN(L)
    T = L.shape[0]
    if not 1 <= lag < T:
        raise ValueError("нужен 1 ≤ lag < T")
    ids = np.unique(L)
    counts = transition_counts(L[:-lag].ravel(), L[lag:].ravel(), ids=ids)
    return (_normalize_rows(counts) if normalize else counts), counts


def shorrocks(P) -> float:
    """Индекс мобильности Шоррокса M = (K − tr P)/(K − 1); 0 — все остаются, 1 — независимость от прошлого.

    K — число строк с наблюдениями (NaN-строки отброшены, чтобы ненаблюдаемый тип не считался «неподвижным»).
    """
    P = np.asarray(P, dtype=float)
    ok = ~np.isnan(P).any(axis=1)
    K = int(ok.sum())
    if K < 2:
        return float("nan")
    return float((K - np.trace(P[np.ix_(ok, ok)])) / (K - 1))


def n_changes(L) -> np.ndarray:
    """Число смен типа у каждого МО (N,)."""
    L = _as_TN(L)
    return (L[1:] != L[:-1]).sum(axis=0)


def stability_share(L, months=None) -> pd.Series:
    """Кривая «выживания»: доля МО без единой смены типа с начала ряда до месяца t включительно.

    Последнее значение — доля МО, ни разу не сменивших тип за весь период.
    """
    L = _as_TN(L)
    changed = np.cumsum(np.vstack([np.zeros((1, L.shape[1]), dtype=bool), L[1:] != L[:-1]]), axis=0) > 0
    return pd.Series(1.0 - changed.mean(axis=1), index=_month_labels(months, L.shape[0]), name="stable_share")


# ----------------------------------------------------------------------------- 2. подтверждённые смены

def confirmed_transitions(L, min_run: int = 3, months=None, confidence=None,
                          confidence_threshold: float = 0.7) -> tuple[np.ndarray, pd.DataFrame]:
    """Подтверждённые смены типа: новый тип засчитывается, если держится ≥ min_run месяцев подряд.

    Иначе МО остаётся в предыдущем подтверждённом типе. Начальный тип — первый отрезок длины ≥ min_run
    (месяцы до него заполняются этим типом; если такого отрезка нет — самый длинный отрезок).
    Хвост короче min_run не засчитывается и попадает в events со status="candidate".
    confidence (T, N) — опционально бутстрэп-уверенность: отрезок подтверждается, только если средняя
    уверенность на его первых min_run месяцах ≥ confidence_threshold.
    events: territory_index, t, month (начало нового типа), from, to, status, run_length,
    confirmed_t / confirmed_month — когда смена становится известна онлайн (t + min_run − 1).
    """
    L = _as_TN(L)
    T, N = L.shape
    mlab = _month_labels(months, T)
    conf = None if confidence is None else np.asarray(confidence, dtype=float).reshape(T, N)
    Lc = np.empty_like(L)
    rows = []
    for i in range(N):
        starts, lens, vals = _runs(L[:, i])
        ok = lens >= min_run
        if conf is not None:
            ok &= np.array([conf[s:s + min(n, min_run), i].mean() >= confidence_threshold
                            for s, n in zip(starts, lens)])
        good = np.flatnonzero(ok)
        j0 = int(good[0]) if good.size else int(np.argmax(lens))
        cur = int(vals[j0])
        Lc[:, i] = cur
        for j in range(j0 + 1, starts.size):
            s, n, v = int(starts[j]), int(lens[j]), int(vals[j])
            if v == cur:
                continue                                    # возврат в подтверждённый тип
            tail = s + n == T
            if ok[j]:
                Lc[s:, i] = v
                ct = s + min_run - 1
                rows.append(dict(territory_index=i, t=s, month=mlab[s], **{"from": cur, "to": v},
                                 status="confirmed", run_length=n, confirmed_t=ct, confirmed_month=mlab[ct]))
                cur = v
            elif tail:
                rows.append(dict(territory_index=i, t=s, month=mlab[s], **{"from": cur, "to": v},
                                 status="candidate", run_length=n, confirmed_t=-1, confirmed_month=None))
    cols = ["territory_index", "t", "month", "from", "to", "status", "run_length", "confirmed_t", "confirmed_month"]
    return Lc, pd.DataFrame(rows, columns=cols)


# ----------------------------------------------------------------------------- 3. сезонность и шум

def seasonal_decomposition_of_changes(L_raw, L_conf, months, min_run: int = 3) -> pd.DataFrame:
    """Разложение сырых смен по месяцам (переход t−1 → t) и итого.

    Каждая сырая смена ровно в одной категории:
      confirmed    — с неё начинается подтверждённый тип (L_conf меняется в t);
      noise_short  — новый тип продержался < min_run и сменился (отменена — шум);
      noise_return — возврат в подтверждённый тип после короткой отлучки (или «разгон» до первого
                     подтверждения) — тоже шум;
      candidate    — хвост ряда короче min_run, ещё не решено.
    Сезонность (ортогонально категориям): is_new_year — пара декабрь→январь; recurrent_12m — та же смена
    (from→to) у того же МО ровно через ±12 месяцев (календарный повтор).
    """
    Lr, Lc = _as_TN(L_raw), _as_TN(L_conf)
    T, N = Lr.shape
    mlab = _month_labels(months, T)
    rem = _remaining_run(Lr)
    raw = Lr[1:] != Lr[:-1]                       # (T−1, N), строка s ↔ переход s → s+1
    conf = (Lc[1:] != Lc[:-1]) & raw
    r = rem[1:]
    tail = (np.arange(1, T)[:, None] + r) == T
    cand = raw & ~conf & (r < min_run) & tail
    short = raw & ~conf & (r < min_run) & ~tail
    ret = raw & ~conf & (r >= min_run)
    rec = np.zeros_like(raw)
    for s in range(T - 1):
        for o in (s - 12, s + 12):
            if 0 <= o < T - 1:
                rec[s] |= raw[s] & raw[o] & (Lr[s] == Lr[o]) & (Lr[s + 1] == Lr[o + 1])
    rows = []
    for s in range(T - 1):
        pair = f"{mlab[s][5:7]}→{mlab[s + 1][5:7]}" if len(mlab[s]) >= 7 else f"{mlab[s]}→{mlab[s + 1]}"
        rows.append(dict(month_from=mlab[s], month_to=mlab[s + 1], cal_pair=pair, is_new_year=pair == NEW_YEAR,
                         n_raw=int(raw[s].sum()), n_confirmed=int(conf[s].sum()), n_noise_short=int(short[s].sum()),
                         n_noise_return=int(ret[s].sum()), n_candidate=int(cand[s].sum()),
                         n_recurrent_12m=int(rec[s].sum())))
    df = pd.DataFrame(rows)
    num = ["n_raw", "n_confirmed", "n_noise_short", "n_noise_return", "n_candidate", "n_recurrent_12m"]
    tot = {c: int(df[c].sum()) for c in num}
    tot.update(month_from="итого", month_to="итого", cal_pair="итого", is_new_year=False)
    df = pd.concat([df, pd.DataFrame([tot])], ignore_index=True)
    df["share_raw"] = df["n_raw"] / N                                   # доля МО, сменивших тип в этот месяц
    df["noise_share"] = (df["n_noise_short"] + df["n_noise_return"]) / df["n_raw"].where(df["n_raw"] > 0)
    df["confirmed_share"] = df["n_confirmed"] / df["n_raw"].where(df["n_raw"] > 0)
    return df


def calendar_share(events: pd.DataFrame, months=None) -> pd.DataFrame:
    """Распределение событий по календарному месяцу начала нового типа.

    expected_share — доля «возможностей» смены в этом календарном месяце (по months[1:], иначе 1/12);
    lift = share / expected_share (> 1 — смены концентрируются в этом месяце, признак сезонности).
    """
    ev = events
    if "status" in ev.columns:
        ev = ev[ev["status"] == "confirmed"]
    cal = ev["month"].map(_cal) if len(ev) else pd.Series([], dtype=int)
    n = cal.value_counts().reindex(range(1, 13), fill_value=0)
    if months is not None:
        exp = pd.Series([_cal(m) for m in list(months)[1:]]).value_counts().reindex(range(1, 13), fill_value=0)
        exp = exp / max(exp.sum(), 1)
    else:
        exp = pd.Series(1 / 12, index=range(1, 13))
    out = pd.DataFrame({"n": n, "share": n / max(int(n.sum()), 1), "expected_share": exp})
    out["lift"] = out["share"] / out["expected_share"].where(out["expected_share"] > 0)
    out.index.name = "cal_month"
    return out


# ----------------------------------------------------------------------------- 4. события Greene

def greene_events(L, threshold: float = 0.3, months=None) -> pd.DataFrame:
    """События жизненного цикла сообществ (Greene, Doyle, Cunningham 2010) между соседними срезами.

    Пара (i из t−1, j из t) сопоставлена, если J(C_i, C_j) ≥ threshold. По числу сопоставлений:
      continue — один-к-одному; merge — ≥ 2 предыдущих → j; split — i → ≥ 2 текущих;
      death — у i нет пары; birth — у j нет пары.
    jaccard: для continue — J пары; merge/split — максимум J по участвующим парам; birth/death — лучший
    (ниже порога) J. size_from / size_to — суммарные размеры участвующих кластеров.
    """
    L = _as_TN(L)
    T = L.shape[0]
    mlab = _month_labels(months, T)
    rows = []

    def add(t, ev, fr, to, j, sf, st):
        rows.append(dict(month=mlab[t], event=ev, from_ids=[int(x) for x in fr], to_ids=[int(x) for x in to],
                         jaccard=float(j), size_from=int(sf), size_to=int(st)))

    for t in range(1, T):
        a, b = L[t - 1], L[t]
        ida, sa = np.unique(a, return_counts=True)
        idb, sb = np.unique(b, return_counts=True)
        J = align.jaccard_matrix(a, b)
        M = J >= threshold
        succ, pred = M.sum(axis=1), M.sum(axis=0)
        for i in range(ida.size):
            if succ[i] == 0:
                add(t, "death", [ida[i]], [], J[i].max(), sa[i], 0)
            elif succ[i] >= 2:
                js = np.flatnonzero(M[i])
                add(t, "split", [ida[i]], idb[js], J[i, js].max(), sa[i], sb[js].sum())
        for j in range(idb.size):
            if pred[j] == 0:
                add(t, "birth", [], [idb[j]], J[:, j].max(), 0, sb[j])
            elif pred[j] >= 2:
                is_ = np.flatnonzero(M[:, j])
                add(t, "merge", ida[is_], [idb[j]], J[is_, j].max(), sa[is_].sum(), sb[j])
        for i, j in zip(*np.nonzero(M)):
            if succ[i] == 1 and pred[j] == 1:
                add(t, "continue", [ida[i]], [idb[j]], J[i, j], sa[i], sb[j])
    cols = ["month", "event", "from_ids", "to_ids", "jaccard", "size_from", "size_to"]
    return pd.DataFrame(rows, columns=cols)


def cluster_sizes_over_time(L, months=None) -> pd.DataFrame:
    """Размеры типов по месяцам: T × K (0 — типа в месяце нет)."""
    L = _as_TN(L)
    ids = np.unique(L)
    S = np.stack([np.bincount(np.searchsorted(ids, row), minlength=ids.size) for row in L])
    return pd.DataFrame(S, index=_month_labels(months, L.shape[0]), columns=ids)


# ----------------------------------------------------------------------------- 5. бутстрэп принадлежности

def _centers(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Центры типов — средние X по меткам: (ids, centers K×F)."""
    ids, inv = np.unique(labels, return_inverse=True)
    C = np.zeros((ids.size, X.shape[1]))
    np.add.at(C, inv.ravel(), X)
    return ids, C / np.bincount(inv.ravel(), minlength=ids.size)[:, None]


def _sqdist(centers: np.ndarray, X: np.ndarray) -> np.ndarray:
    return ((X * X).sum(1)[:, None] - 2.0 * X @ centers.T + (centers * centers).sum(1)[None, :]).clip(min=0)


def _bootstrap_one(X, A, k, n_boot, frac, seed, ref, run_fn) -> tuple[np.ndarray, np.ndarray]:
    N = X.shape[0]
    ref = np.asarray(run_fn(X, A, k, seed) if ref is None else ref).astype(np.int64)
    ids = np.unique(ref)
    votes = np.zeros((N, ids.size))
    rng = np.random.default_rng(seed)
    m = min(N, max(k + 1, int(round(frac * N))))
    for b in range(n_boot):
        idx = np.sort(rng.choice(N, size=m, replace=False))
        A_sub = None if A is None else A[idx][:, idx]
        lab_sub = np.asarray(run_fn(X[idx], A_sub, k, seed + b + 1)).astype(np.int64)
        cid, C = _centers(X[idx], lab_sub)
        lab = cid[np.argmin(_sqdist(C, X), axis=1)]          # МО вне подвыборки — к ближайшему центру
        lab[idx] = lab_sub
        lab = align.align_to_reference(ref, lab)
        pos = np.searchsorted(ids, lab)
        hit = (pos < ids.size) & (ids[np.minimum(pos, ids.size - 1)] == lab)   # несопоставленные типы — без голоса
        votes[np.flatnonzero(hit), pos[hit]] += 1
    return ids, votes / n_boot


def bootstrap_membership(X, A_list, method: str | None, k: int, n_boot: int = 30, frac: float = 0.8, seed: int = 0,
                         ref_labels=None, run_fn: Callable | None = None, params: dict | None = None):
    """Устойчивость типа МО: подвыборки МО (frac, без возвращения), перекластеризация тем же методом,
    выравнивание к ref_labels (align_to_reference), накопление частот.

    run_fn(X_sub, A_sub, k, seed) -> labels; по умолчанию methods.run_method(method, …, params).
    Один срез: X (N, F), A_list — csr (или None) → prob (N, K), confidence (N,).
    Несколько: X (T, N, F), A_list — список из T → prob (T, N, K), confidence (T, N).
    Столбцы prob — отсортированные метки ref_labels (для нескольких срезов — их объединение).
    confidence = доля голосов за модальный тип. Голос МО вне подвыборки — по ближайшему центру (по X).
    """
    if run_fn is None:
        from pulsar import methods

        def run_fn(Xs, As, kk, s):
            return methods.run_method(method, Xs, As, kk, seed=s, params=dict(params or {}))

    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        _, prob = _bootstrap_one(X, A_list, k, n_boot, frac, seed, ref_labels, run_fn)
        return prob, prob.max(axis=1)
    T, N, _ = X.shape
    A_list = [None] * T if A_list is None else A_list
    refs = [None] * T if ref_labels is None else list(np.asarray(ref_labels))
    parts = [_bootstrap_one(X[t], A_list[t], k, n_boot, frac, seed + 1000 * t, refs[t], run_fn) for t in range(T)]
    all_ids = np.unique(np.concatenate([p[0] for p in parts]))
    prob = np.zeros((T, N, all_ids.size))
    for t, (ids, pr) in enumerate(parts):
        prob[t][:, np.searchsorted(all_ids, ids)] = pr
    return prob, prob.max(axis=2)


# ----------------------------------------------------------------------------- 6. out-of-time

def centroid_classifier(centers, X_t) -> np.ndarray:
    """Номер ближайшего (евклид) центра — позиция строки в centers."""
    return np.argmin(_sqdist(np.asarray(centers, float), np.asarray(X_t, float)), axis=1)


def margin(centers, X_t) -> np.ndarray:
    """Относительный зазор (d₂ − d₁)/d₁ между расстояниями до 2-го и 1-го ближайших центров (N,)."""
    D = np.sqrt(np.sort(_sqdist(np.asarray(centers, float), np.asarray(X_t, float)), axis=1))
    return (D[:, 1] - D[:, 0]) / np.maximum(D[:, 0], 1e-12)


def _event_keys(ev: pd.DataFrame, t_min: int) -> set:
    ev = ev[(ev["status"] == "confirmed") & (ev["t"] >= t_min)]
    return set(zip(ev["territory_index"], ev["to"]))


def out_of_time(X, L, train_idx, test_idx, method_centers_fn: Callable | None = None, k: int | None = None,
                min_run: int = 3, months=None) -> dict:
    """Честная онлайн-проверка: центры типов — по обучающим месяцам, тест классифицируется без пересчёта.

    Центры = средние X по типу L, пул по всем (t ∈ train_idx, МО); method_centers_fn(X_train, L_train, k) ->
    (ids, centers) позволяет взять центры самого метода (например, KEFRiN centers_X).
    Возвращает: pred (T_test, N); ari / agreement по тестовым месяцам против полной разметки L;
    transition_last — переходы L[последний train] → pred[последний test] и его Шоррокс; confirmed —
    полнота/точность подтверждённых смен теста онлайн против полной разметки (ключ: МО + новый тип;
    exact_month_recall — с совпадением месяца начала); margin_last — зазор в последнем тестовом месяце.
    """
    X = np.asarray(X, dtype=float)
    L = _as_TN(L)
    train_idx, test_idx = np.asarray(train_idx), np.asarray(test_idx)
    Xtr, Ltr = X[train_idx], L[train_idx]
    if method_centers_fn is not None:
        ids, C = method_centers_fn(Xtr, Ltr, k)
        ids, C = np.asarray(ids), np.asarray(C, float)
    else:
        ids, C = _centers(Xtr.reshape(-1, X.shape[2]), Ltr.ravel())
    pred = np.stack([ids[centroid_classifier(C, X[t])] for t in test_idx])
    full = L[test_idx]
    ari = [float(adjusted_rand_score(full[s], pred[s])) for s in range(len(test_idx))]
    agr = [float((full[s] == pred[s]).mean()) for s in range(len(test_idx))]
    mlab = _month_labels(months, L.shape[0])
    P_last, C_last = transition_matrix(np.vstack([L[train_idx[-1]], pred[-1]]))
    # подтверждённые смены на склейке «train (полные метки) + test (онлайн)» против полной разметки
    seq = np.r_[train_idx, test_idx]
    seq_months = [mlab[t] for t in seq]
    _, ev_on = confirmed_transitions(np.vstack([Ltr, pred]), min_run=min_run, months=seq_months)
    _, ev_full = confirmed_transitions(L[seq], min_run=min_run, months=seq_months)
    t0 = len(train_idx)
    k_on, k_full = _event_keys(ev_on, t0), _event_keys(ev_full, t0)
    ex_on = set(zip(*[ev_on.loc[(ev_on.status == "confirmed") & (ev_on.t >= t0), c] for c in
                      ("territory_index", "to", "t")]))
    ex_full = set(zip(*[ev_full.loc[(ev_full.status == "confirmed") & (ev_full.t >= t0), c] for c in
                        ("territory_index", "to", "t")]))
    return dict(
        ids=ids, centers=C, pred=pred,
        test_months=[mlab[t] for t in test_idx],
        ari=ari, ari_mean=float(np.mean(ari)), agreement=agr, agreement_mean=float(np.mean(agr)),
        transition_last=P_last, transition_last_counts=C_last, shorrocks_last=shorrocks(P_last),
        confirmed=dict(n_full=len(k_full), n_online=len(k_on), n_both=len(k_full & k_on),
                       recall=len(k_full & k_on) / len(k_full) if k_full else float("nan"),
                       precision=len(k_full & k_on) / len(k_on) if k_on else float("nan"),
                       exact_month_recall=len(ex_full & ex_on) / len(ex_full) if ex_full else float("nan")),
        events_online=ev_on, margin_last=margin(C, X[test_idx[-1]]),
    )


# ----------------------------------------------------------------------------- 7–8. таблицы и сводка

def trajectory_table(L_conf, months, meta: pd.DataFrame | None = None, id_col: str = "territory_id",
                     sep: str = ",") -> pd.DataFrame:
    """Траектории для лендинга: строка meta i ↔ столбец i в L_conf.

    territory_id (из meta[id_col] или индекс), trajectory — типы по месяцам через sep, n_changes,
    first_type, last_type, last_change_month (None, если смен не было); прочие столбцы meta — следом.
    """
    L = _as_TN(L_conf)
    T, N = L.shape
    mlab = _month_labels(months, T)
    ch = L[1:] != L[:-1]
    any_ch = ch.any(axis=0)
    last_t = T - 1 - np.argmax(ch[::-1], axis=0)               # индекс перехода s → месяц s+1
    if meta is None:
        tid = np.arange(N)
        rest = pd.DataFrame(index=range(N))
    else:
        if len(meta) != N:
            raise ValueError("len(meta) != N")
        tid = meta[id_col].to_numpy() if id_col in meta.columns else meta.index.to_numpy()
        rest = meta.drop(columns=[id_col], errors="ignore")
    out = pd.DataFrame({
        "territory_id": tid,
        "trajectory": [sep.join(map(str, L[:, i])) for i in range(N)],
        "n_changes": ch.sum(axis=0),
        "first_type": L[0],
        "last_type": L[-1],
        "last_change_month": [mlab[last_t[i]] if any_ch[i] else None for i in range(N)],
    })
    return pd.concat([out, rest.reset_index(drop=True)], axis=1)


def _py(x):
    """Привести к JSON-совместимым типам."""
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.ndarray):
        return _py(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def summarize(L_raw, L_conf, months, min_run: int = 3, greene_threshold: float = 0.3) -> dict:
    """JSON-сериализуемая сводка динамики (сырая и подтверждённая разметки)."""
    Lr, Lc = _as_TN(L_raw), _as_TN(L_conf)
    T, N = Lr.shape

    def mob(L):
        return {f"lag{g}": shorrocks(transition_matrix(L, lag=g)[0]) if g < T else None for g in (1, 12)}

    dec = seasonal_decomposition_of_changes(Lr, Lc, months, min_run=min_run)
    tot = dec.iloc[-1]
    body = dec.iloc[:-1]
    n_raw = int(tot["n_raw"])
    gr, gc = greene_events(Lr, greene_threshold, months), greene_events(Lc, greene_threshold, months)
    sizes = cluster_sizes_over_time(Lc, months)
    return _py(dict(
        K=int(np.unique(Lr).size), K_confirmed=int(np.unique(Lc).size), n_territories=N, n_months=T,
        months=[str(m) for m in months],
        shorrocks=dict(raw=mob(Lr), confirmed=mob(Lc)),
        stable_share=dict(raw=float(stability_share(Lr).iloc[-1]), confirmed=float(stability_share(Lc).iloc[-1])),
        n_changes=dict(raw=n_raw, confirmed=int(tot["n_confirmed"]),
                       territories_with_confirmed=int((n_changes(Lc) > 0).sum())),
        changes=dict(
            noise_share=(int(tot["n_noise_short"]) + int(tot["n_noise_return"])) / n_raw if n_raw else None,
            confirmed_share=int(tot["n_confirmed"]) / n_raw if n_raw else None,
            candidate=int(tot["n_candidate"]),
            seasonal_new_year_share=int(body.loc[body.is_new_year, "n_raw"].sum()) / n_raw if n_raw else None,
            seasonal_recurrent_share=int(tot["n_recurrent_12m"]) / n_raw if n_raw else None,
            by_month={r.month_to: dict(raw=int(r.n_raw), confirmed=int(r.n_confirmed))
                      for r in body.itertuples()},
        ),
        sizes={str(c): sizes[c].tolist() for c in sizes.columns},
        greene=dict(raw=gr["event"].value_counts().to_dict(), confirmed=gc["event"].value_counts().to_dict(),
                    threshold=greene_threshold),
    ))
