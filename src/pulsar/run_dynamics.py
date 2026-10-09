"""Конвейер динамики итоговой типологии. `PYTHONPATH=src python -m pulsar.run_dynamics [--method M --k K]`.

Метод/K/слой/опорный месяц — configs/final.yaml (аргументы командной строки переопределяют),
параметры динамики — configs/dynamics.yaml. Выход — outputs/dynamics/ (см. write-блоки ниже).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pulsar import align, build_graphs, data, dynamics as dyn, features, run_cluster

OUT = data.ROOT / "outputs" / "dynamics"
GRAPH_ONLY = {"leiden", "spectral"}          # методы без X: центры для голосов вне подвыборки — приближение


# ----------------------------------------------------------------------------- метки

def renumber_by_reference(L: np.ndarray, t_ref: int) -> np.ndarray:
    """Перенумеровать сквозные метки: типы опорного месяца → 0..K−1 по убыванию размера в нём,
    прочие (появляющиеся только в других месяцах) — следом по убыванию общего размера."""
    ids_ref, cnt_ref = np.unique(L[t_ref], return_counts=True)
    order = list(ids_ref[np.lexsort((ids_ref, -cnt_ref))])
    ids_all, cnt_all = np.unique(L, return_counts=True)
    rest = [i for i in ids_all[np.lexsort((ids_all, -cnt_all))] if i not in set(order)]
    mapping = {int(old): new for new, old in enumerate(order + rest)}
    return np.vectorize(mapping.get)(L).astype(np.int64)


def through_labels(L: np.ndarray, method: str, t_ref: int) -> np.ndarray:
    """Сквозные номера: помесячные методы — align_sequence (Жаккар, венгерский), temporal Leiden — как есть."""
    if method != "temporal_leiden":
        L = np.vstack(align.align_sequence(list(L)))
    return renumber_by_reference(np.asarray(L, dtype=np.int64), t_ref)


# ----------------------------------------------------------------------------- бутстрэп

def _boot_month(X_t, A_t, ref_t, method, k, params, n_boot, frac, seed):
    prob, conf = dyn.bootstrap_membership(X_t, A_t, method, k, n_boot=n_boot, frac=frac, seed=seed,
                                          ref_labels=ref_t, params=params)
    return prob, conf


def bootstrap_all(X, A_list, L, method, k, params, bcfg, seed, n_jobs):
    """Бутстрэп по всем месяцам параллельно → (prob_assigned (T,N), confidence (T,N))."""
    T, N = L.shape
    res = Parallel(n_jobs=n_jobs)(delayed(_boot_month)(X[t], A_list[t], L[t], method, k, params, bcfg["n_boot"],
                                                       bcfg["frac"], seed + 1000 * t) for t in range(T))
    p_assigned = np.empty((T, N))
    conf = np.empty((T, N))
    for t, (prob, c) in enumerate(res):
        ids = np.unique(L[t])                                  # столбцы prob — отсортированные метки ref
        p_assigned[t] = prob[np.arange(N), np.searchsorted(ids, L[t])]
        conf[t] = c
    return p_assigned, conf


# ----------------------------------------------------------------------------- запись

def _matrix_long(counts: pd.DataFrame) -> pd.DataFrame:
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    out = counts.stack().rename("count").to_frame().join(shares.stack().rename("share_row"))
    out.index.names = ["from", "to"]
    return out.reset_index()


def _greene_csv(L, thr, months, tag) -> pd.DataFrame:
    g = dyn.greene_events(L, thr, months)
    for c in ("from_ids", "to_ids"):
        g[c] = g[c].map(lambda v: ";".join(map(str, v)))
    g.insert(0, "labels", tag)
    return g


def main(argv=None) -> dict:
    fin = data.load_config("final")
    dcfg = data.load_config("dynamics")
    mcfg = data.load_config("methods")
    ap = argparse.ArgumentParser(description="Динамика итоговой типологии")
    ap.add_argument("--method", default=fin["method"])
    ap.add_argument("--k", type=int, default=fin["k"])
    ap.add_argument("--layer", default=fin["layer"])
    ap.add_argument("--reference-month", default=fin["reference_month"])
    ap.add_argument("--n-boot", type=int, default=dcfg["bootstrap"]["n_boot"])
    ap.add_argument("--n-jobs", type=int, default=dcfg.get("bootstrap_n_jobs", -1))
    ap.add_argument("--skip-bootstrap", action="store_true", help="без бутстрэпа (подтверждение только по min_run)")
    args = ap.parse_args(argv)
    t_start = time.time()
    timing = {}

    f = features.load()
    X, months, ids, panel = f["X"].astype(np.float64), list(f["months"]), f["ids"], f["panel"]
    meta = panel[panel["in_panel"]].reset_index(drop=True)
    assert np.array_equal(meta["territory_id"].to_numpy(), ids), "порядок panel[in_panel] ≠ ids"
    t_ref = months.index(args.reference_month)
    L_in = run_cluster.load_labels(args.method, args.k)
    assert L_in.shape == X.shape[:2], f"метки {L_in.shape} ≠ X {X.shape[:2]}"
    L = through_labels(L_in, args.method, t_ref)

    min_run, thr = int(dcfg["min_run"]), float(dcfg["greene_threshold"])
    bcfg = dict(dcfg["bootstrap"], n_boot=args.n_boot)
    seed = int(dcfg.get("seed", 0))

    # бутстрэп-уверенность по срезам
    boot_note = None
    p_assigned = conf = None
    if args.skip_bootstrap or args.method in dcfg.get("bootstrap_skip", []):
        boot_note = "бутстрэп пропущен" + (" (temporal Leiden: помесячная перекластеризация не воспроизводит метод)"
                                          if args.method == "temporal_leiden" else "")
    else:
        t0 = time.time()
        A_list = build_graphs.load_layer(args.layer, months)
        params = dict(mcfg["params"].get(args.method, {}))
        p_assigned, conf = bootstrap_all(X, A_list, L, args.method, args.k, params, bcfg, seed, args.n_jobs)
        timing["bootstrap_s"] = round(time.time() - t0, 1)
        if args.method in GRAPH_ONLY:
            boot_note = "метод только по графу: голоса МО вне подвыборки — по ближайшему центру в X (приближение)"

    use_conf = bool(dcfg.get("confirm_with_confidence", True)) and p_assigned is not None
    Lc, events = dyn.confirmed_transitions(L, min_run=min_run, months=months,
                                           confidence=p_assigned if use_conf else None,
                                           confidence_threshold=float(dcfg["confidence_threshold"]))
    events["territory_id"] = ids[events["territory_index"].to_numpy()]

    # out-of-time: центры по 2023 → онлайн 2024
    tr = dyn.month_indices(months, *dcfg["train_months"])
    te = dyn.month_indices(months, *dcfg["test_months"])
    t0 = time.time()
    oot = dyn.out_of_time(X, L, tr, te, min_run=min_run, months=months)
    timing["out_of_time_s"] = round(time.time() - t0, 1)

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "labels_raw.npy", L.astype(np.int32))
    np.save(OUT / "labels_confirmed.npy", Lc.astype(np.int32))
    if conf is not None:
        np.save(OUT / "confidence.npy", conf.astype(np.float32))              # доля голосов за модальный тип
        np.save(OUT / "prob_assigned.npy", p_assigned.astype(np.float32))     # доля голосов за назначенный тип
    events.to_csv(OUT / "events_confirmed.csv", index=False)
    pd.concat([_greene_csv(L, thr, months, "raw"), _greene_csv(Lc, thr, months, "confirmed")]).to_csv(
        OUT / "greene_events.csv", index=False)
    for lag in (1, 12):
        dyn.transition_matrix(L, lag=lag)[0].to_csv(OUT / f"transition_matrix_lag{lag}.csv")
        dyn.transition_matrix(Lc, lag=lag)[0].to_csv(OUT / f"transition_matrix_lag{lag}_confirmed.csv")
    i_a, i_b = months.index("2023-12"), months.index("2024-12")
    cnt_ab = dyn.transition_counts(Lc[i_a], Lc[i_b], ids=np.unique(Lc))
    _matrix_long(cnt_ab).to_csv(OUT / "transitions_2023-12_to_2024-12.csv", index=False)
    dyn.cluster_sizes_over_time(Lc, months).to_csv(OUT / "sizes_over_time.csv", index_label="month")
    dyn.cluster_sizes_over_time(L, months).to_csv(OUT / "sizes_over_time_raw.csv", index_label="month")
    dec = dyn.seasonal_decomposition_of_changes(L, Lc, months, min_run=min_run)
    dec.to_csv(OUT / "change_decomposition.csv", index=False)
    dyn.calendar_share(events, months).to_csv(OUT / "calendar_share.csv")

    traj = dyn.trajectory_table(Lc, months, meta)
    traj.insert(3, "n_changes_raw", dyn.n_changes(L))
    if conf is not None:
        traj.insert(4, "confidence_last", conf[-1])
        traj.insert(5, "confidence_mean", conf.mean(axis=0))
    traj.to_parquet(OUT / "trajectories.parquet", index=False)

    oot_json = dyn._py(dict(
        train_months=[months[t] for t in tr], test_months=oot["test_months"],
        ari_by_month=dict(zip(oot["test_months"], oot["ari"])), ari_mean=oot["ari_mean"],
        accuracy_by_month=dict(zip(oot["test_months"], oot["agreement"])), accuracy_mean=oot["agreement_mean"],
        confirmed_transitions_2024=oot["confirmed"], shorrocks_2023_12_to_2024_12_online=oot["shorrocks_last"],
        transition_2023_12_to_2024_12_online={str(i): {str(j): int(v) for j, v in row.items()}
                                              for i, row in oot["transition_last_counts"].iterrows()},
        margin_last=dict(median=float(np.median(oot["margin_last"])), share_below_0_1=float((oot["margin_last"] < 0.1).mean())),
        type_ids=oot["ids"], centers_2023=oot["centers"],
    ))
    (OUT / "out_of_time.json").write_text(json.dumps(oot_json, ensure_ascii=False, indent=2))

    summ = dyn.summarize(L, Lc, months, min_run=min_run, greene_threshold=thr)
    P_ab = cnt_ab.div(cnt_ab.sum(axis=1).replace(0, np.nan), axis=0)
    summ.update(dyn._py(dict(
        params=dict(method=args.method, k=args.k, layer=args.layer, reference_month=args.reference_month,
                    min_run=min_run, greene_threshold=thr, bootstrap=bcfg, confirm_with_confidence=use_conf,
                    confidence_threshold=dcfg["confidence_threshold"], train_months=dcfg["train_months"],
                    test_months=dcfg["test_months"], seed=seed),
        shorrocks_2023_12_to_2024_12_confirmed=dyn.shorrocks(P_ab),
        out_of_time=dict(ari_mean=oot["ari_mean"], accuracy_mean=oot["agreement_mean"],
                         confirmed_recall=oot["confirmed"]["recall"], confirmed_precision=oot["confirmed"]["precision"],
                         shorrocks_online=oot["shorrocks_last"]),
        bootstrap=None if conf is None else dict(
            confidence_median=float(np.median(conf)), share_confident=float((conf >= dcfg["confidence_threshold"]).mean()),
            prob_assigned_median=float(np.median(p_assigned))),
        bootstrap_note=boot_note,
        n_candidates=int((events["status"] == "candidate").sum()),
        timing=dict(timing, total_s=round(time.time() - t_start, 1)),
    )))
    (OUT / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))

    sh, ch = summ["shorrocks"], summ["changes"]
    fmt = lambda v: "—" if v is None else f"{v:.3f}"   # noqa: E731
    print(f"метод {args.method}, K={args.k}, слой {args.layer}, опорный месяц {args.reference_month}; "
          f"типов сквозных: {summ['K']} (подтв. {summ['K_confirmed']})")
    print(f"доля стабильных МО: сырая {summ['stable_share']['raw']:.3f}, подтверждённая {summ['stable_share']['confirmed']:.3f}")
    print(f"смен: сырых {summ['n_changes']['raw']}, подтверждённых {summ['n_changes']['confirmed']} "
          f"(у {summ['n_changes']['territories_with_confirmed']} МО), кандидатов {summ['n_candidates']}"
          f"{' [подтверждение с бутстрэпом]' if use_conf else ''}")
    print(f"Шоррокс lag1: сырой {fmt(sh['raw']['lag1'])} / подтв. {fmt(sh['confirmed']['lag1'])}; "
          f"lag12: сырой {fmt(sh['raw']['lag12'])} / подтв. {fmt(sh['confirmed']['lag12'])}")
    print(f"out-of-time 2024: ARI {oot['ari_mean']:.3f}, accuracy {oot['agreement_mean']:.3f}, "
          f"полнота/точность подтв. смен {fmt(oot['confirmed']['recall'])}/{fmt(oot['confirmed']['precision'])}")
    print(f"доля шума среди сырых смен {fmt(ch['noise_share'])}; сезонных: дек→янв {fmt(ch['seasonal_new_year_share'])}, "
          f"повтор через 12 мес. {fmt(ch['seasonal_recurrent_share'])}")
    if conf is not None:
        print(f"бутстрэп: медиана уверенности {np.median(conf):.3f}, доля ≥ {dcfg['confidence_threshold']}: "
              f"{(conf >= dcfg['confidence_threshold']).mean():.3f}, {timing['bootstrap_s']} с")
    if boot_note:
        print("примечание:", boot_note)
    print(f"готово за {time.time() - t_start:.1f} с → {OUT}")
    return summ


if __name__ == "__main__":
    main()
