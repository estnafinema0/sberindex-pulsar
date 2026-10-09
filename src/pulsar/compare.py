"""Сравнение методов и абляция правил ребра. `python -m pulsar.compare`.

Входы: outputs/cluster/metrics_long.csv и метки (run_cluster), графы (build_graphs), признаки (features).
Выходы (outputs/compare/):
  fixed_k.csv            — медиана/IQR индексов по 24 мес. при K = k_fixed
  best_k.csv             — лучшее K каждого метода (по консенсусу z-оценок) и индексы при нём
  ranking_fixed_k.csv    — Борда, пороговое правило Алескерова, частота лидерства в бутстрэпе
  paired_bootstrap.csv   — ДИ разностей индексов «метод A − метод B» (парный бутстрэп по месяцам)
  perm_null.csv          — z против перестановок меток (все методы, все месяцы) при K = k_fixed
  config_null.csv        — z сетевых индексов против configuration model
  ablation_layers.csv    — ICVI одного метода на разных слоях графа; ablation_ari.csv — ARI типологий между слоями
  k_selection.csv        — консенсус по K
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from pulsar import build_graphs, data, evaluate, features, run_cluster

OUT = data.ROOT / "outputs" / "compare"
MAIN = ["SW", "CH", "S_Dbw", "AVI", "AVU", "MQ"]          # шесть индексов из Положения
EXTRA = ["ANUI", "DBI", "CH_N", "Qdens"]


def _long() -> pd.DataFrame:
    """Основные прогоны: metrics_long.csv (помесячные методы) + metrics_long_temporal.csv (temporal Leiden)."""
    skip = ("_behaviour", "_comovement", "_gravity")  # файлы абляции слоёв
    files = [f for f in sorted(run_cluster.OUT.glob("metrics_long*.csv")) if not any(f.stem.endswith(sfx) for sfx in skip)]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def fixed_k_table(long: pd.DataFrame, k: int) -> pd.DataFrame:
    sub = long[long.K == k]
    return evaluate.summarize_over_months(sub, by=("method",), metrics=MAIN + EXTRA).round(4)


def perm_null_all(k: int, n_perm: int, seed: int, months: list[str], X, A_list) -> pd.DataFrame:
    """z-оценки против перестановок меток для каждого метода и месяца при K = k."""
    rows = []
    for name in METHOD_ORDER:
        try:
            L = run_cluster.load_labels(name, k)
        except FileNotFoundError:
            continue
        for t, m in enumerate(months):
            r = evaluate.permutation_null(X[t], A_list[t], L[t], n_perm=n_perm, seed=seed + t, metrics=MAIN + EXTRA)
            r["method"], r["month"], r["K"] = name, m, k
            rows.append(r)
    return pd.concat(rows, ignore_index=True)


def k_selection(long: pd.DataFrame, znull: pd.DataFrame) -> pd.DataFrame:
    """Выбор K — явное правило, а не один индекс (ESWA 2026: «небольшая взаимодополняющая панель»).

    Для каждого метода и K считаем средние по месяцам ориентированные z-оценки. Индексы CH, AVI, MQ, ANUI растут с K
    почти монотонно (больше кластеров — больше «структуры» против перестановки), поэтому сами по себе K не выбирают.
    Балл K = сумма трёх нормированных критериев: (1) z_AVU — единственный сетевой индекс с K-зависимым случайным
    уровнем, у которого есть пик; (2) z_SW — компактность/разделимость по признакам; (3) бутстрэп-устойчивость
    разбиения (ARI между подвыборками, outputs/compare/k_stability.csv), если посчитана. Каждый критерий
    масштабируется в [0, 1] внутри метода; лучшее K — максимум суммы. Таблица сохраняет все компоненты, чтобы
    правило было проверяемо.
    """
    z = znull.groupby(["method", "K", "metric"]).z.mean().unstack("metric")
    out = z.copy()
    stab_path = OUT / "k_stability.csv"
    stab = pd.read_csv(stab_path).groupby(["method", "K"]).ari_median.mean() if stab_path.exists() else None
    norm = lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0 + 0.5
    scores = []
    for m, g in out.groupby(level="method"):
        sc = norm(g["AVU"]) + norm(g["SW"])
        if stab is not None and m in stab.index.get_level_values(0):
            st = stab.loc[m].reindex(g.index.get_level_values("K")).to_numpy()
            if not np.isnan(st).all():
                sc = sc + norm(pd.Series(st, index=g.index).fillna(np.nanmin(st)))
                out.loc[g.index, "stability_ari"] = st
        scores.append(sc)
    out["k_score"] = pd.concat(scores)
    best = out.groupby(level="method").k_score.idxmax()
    out["best"] = [idx in set(best.values) for idx in out.index]
    return out.reset_index()


def best_k_table(long: pd.DataFrame, ksel: pd.DataFrame) -> pd.DataFrame:
    best = ksel[ksel.best][["method", "K"]]
    sub = long.merge(best, on=["method", "K"])
    t = evaluate.summarize_over_months(sub, by=("method", "K"), metrics=MAIN + EXTRA).round(4)
    return t


def ranking(long: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = long[long.K == k]
    med = sub.groupby("method")[MAIN].median()
    b = evaluate.borda(med, MAIN).rename("borda")
    thr = evaluate.threshold_aggregation(med, MAIN, grades=3)
    # частота лидерства по бутстрэпу месяцев: по каждому индексу, затем доля индексов, где метод лидер
    wins = {}
    for m in MAIN:
        piv = sub.pivot_table(index="month", columns="method", values=m) * evaluate.DIRECTION[m]
        wins[m] = evaluate.bootstrap_win_rate(piv, n_boot=1000, seed=0)
    wins = pd.DataFrame(wins)
    wins["mean_win_rate"] = wins.mean(axis=1)
    out = med.join(b).join(thr[["n1", "n2", "n3", "rank"]].rename(columns={"rank": "threshold_rank"})).join(wins["mean_win_rate"])
    return out.sort_values("borda", ascending=False), wins


def paired(long: pd.DataFrame, k: int) -> pd.DataFrame:
    sub = long[long.K == k]
    rows = []
    for m in MAIN + ["ANUI"]:
        piv = sub.pivot_table(index="month", columns="method", values=m) * evaluate.DIRECTION[m]
        r = evaluate.paired_bootstrap(piv, n_boot=2000, seed=0)
        r["metric"] = m
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def config_null(k: int, months: list[str], A_list, n_rep: int, seed: int, sample_months: list[str]) -> pd.DataFrame:
    rows = []
    for name in METHOD_ORDER:
        try:
            L = run_cluster.load_labels(name, k)
        except FileNotFoundError:
            continue
        for m in sample_months:
            t = months.index(m)
            r = evaluate.configuration_model_null(A_list[t], L[t], n_rep=n_rep, seed=seed)
            r["method"], r["month"] = name, m
            rows.append(r)
    return pd.concat(rows, ignore_index=True)


def ablation(k: int, layers: list[str], methods_for_ablation: list[str], months: list[str], X) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Те же методы на разных слоях: индексы и ARI типологий между слоями (по месяцам, медиана)."""
    longs = []
    labels = {}
    for layer in layers:
        tag = f"_{layer}"
        try:
            lg = pd.read_csv(run_cluster.OUT / f"metrics_long{tag}.csv")
        except FileNotFoundError:
            lg = run_cluster.run(layer=layer, methods_subset=methods_for_ablation, k_grid=[k], tag=tag)
        longs.append(lg)
        for name in methods_for_ablation:
            labels[(layer, name)] = run_cluster.load_labels(name, k, tag=tag)
    # основной слой уже посчитан без тега
    main_layer = data.load_config("graph")["primary"]
    lg_main = _long(); lg_main = lg_main[(lg_main.K == k) & lg_main.method.isin(methods_for_ablation)]
    longs.append(lg_main)
    for name in methods_for_ablation:
        labels[(main_layer, name)] = run_cluster.load_labels(name, k)
    long = pd.concat(longs, ignore_index=True)
    table = evaluate.summarize_over_months(long, by=("layer", "method"), metrics=MAIN + EXTRA).round(4)
    rows = []
    all_layers = layers + [main_layer]
    for name in methods_for_ablation:
        for a, b in itertools.combinations(all_layers, 2):
            La, Lb = labels[(a, name)], labels[(b, name)]
            aris = [adjusted_rand_score(La[t], Lb[t]) for t in range(len(months))]
            rows.append({"method": name, "layer_a": a, "layer_b": b, "ari_median": float(np.median(aris)), "ari_iqr": float(np.subtract(*np.percentile(aris, [75, 25])))})
    return table, pd.DataFrame(rows)


METHOD_ORDER = ["kmeans", "ward", "spectral", "leiden", "temporal_leiden", "kefrin", "kefrin_balanced"]


def main(cfg: dict | None = None) -> None:
    cfg = cfg or data.load_config("compare")
    mcfg = data.load_config("methods")
    k = mcfg["k_fixed"]
    OUT.mkdir(parents=True, exist_ok=True)
    f = features.load(); X, months = f["X"].astype(np.float64), f["months"]
    layer = data.load_config("graph")["primary"]
    A_list = build_graphs.load_layer(layer, months)
    long = _long()

    fixed_k_table(long, k).to_csv(OUT / "fixed_k.csv")
    # перестановочные z для всех K (нужны для выбора K) — на подвыборке месяцев, чтобы уложиться по времени
    rows = []
    for kk in mcfg["k_grid"]:
        for name in METHOD_ORDER:
            try:
                L = run_cluster.load_labels(name, kk)
            except FileNotFoundError:
                continue
            for m in cfg["perm"]["months_for_k"]:
                t = months.index(m)
                r = evaluate.permutation_null(X[t], A_list[t], L[t], n_perm=cfg["perm"]["n_perm_k"], seed=cfg["seed"] + t, metrics=MAIN + EXTRA)
                r["method"], r["month"], r["K"] = name, m, kk
                rows.append(r)
    zk = pd.concat(rows, ignore_index=True); zk.to_csv(OUT / "perm_null_by_k.csv", index=False)
    ksel = k_selection(long, zk); ksel.to_csv(OUT / "k_selection.csv", index=False)
    best_k_table(long, ksel).to_csv(OUT / "best_k.csv")

    znull = perm_null_all(k, cfg["perm"]["n_perm"], cfg["seed"], months, X, A_list); znull.to_csv(OUT / "perm_null.csv", index=False)
    rank, wins = ranking(long, k); rank.to_csv(OUT / "ranking_fixed_k.csv"); wins.to_csv(OUT / "win_rates.csv")
    paired(long, k).to_csv(OUT / "paired_bootstrap.csv", index=False)
    config_null(k, months, A_list, cfg["config_null"]["n_rep"], cfg["seed"], cfg["config_null"]["months"]).to_csv(OUT / "config_null.csv", index=False)
    table, ari = ablation(k, cfg["ablation"]["layers"], cfg["ablation"]["methods"], months, X)
    table.to_csv(OUT / "ablation_layers.csv"); ari.to_csv(OUT / "ablation_ari.csv", index=False)
    print(rank.round(3).to_string())


if __name__ == "__main__":
    main()
