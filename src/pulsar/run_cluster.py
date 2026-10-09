"""Запуск всех методов на одинаковых X_t и A_t по сетке K. `python -m pulsar.run_cluster`.

Выход: outputs/cluster/labels_{method}_K{k}.npy — массив (T, N) меток (для temporal Leiden — сквозные номера),
outputs/cluster/metrics_long.csv — панель ICVI по (месяц, метод, K), outputs/cluster/timing.csv.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from pulsar import build_graphs, data, evaluate, features, methods

OUT = data.ROOT / "outputs" / "cluster"


def run(cfg: dict | None = None, layer: str | None = None, months_subset: list[str] | None = None,
        methods_subset: list[str] | None = None, k_grid: list[int] | None = None, tag: str = "") -> pd.DataFrame:
    cfg = cfg or data.load_config("methods")
    gcfg = data.load_config("graph")
    layer = layer or gcfg["primary"]
    f = features.load()
    X, months = f["X"].astype(np.float64), f["months"]
    if months_subset:
        idx = [months.index(m) for m in months_subset]
        X, months = X[idx], months_subset
    A_list = build_graphs.load_layer(layer, months)
    names = methods_subset or (list(cfg["methods"]) + ["temporal_leiden"])
    k_grid = k_grid or cfg["k_grid"]
    seed = cfg["seed"]
    OUT.mkdir(parents=True, exist_ok=True)
    rows, timing = [], []
    for k in k_grid:
        for name in names:
            t0 = time.time()
            if name == "temporal_leiden":
                tl = cfg["temporal_leiden"]
                extra = {key: tl[key] for key in ("gamma_lo", "gamma_hi", "max_iter", "n_iterations") if key in tl}
                labs = methods.run_temporal_leiden(A_list, k, interslice_weight=tl["interslice_weight"],
                                                   seed=seed, min_size=tl["min_size"], **extra)
                L = np.vstack(labs)
            else:
                L = np.vstack([methods.run_method(name, X[t], A_list[t], k, seed=seed, params=dict(cfg["params"].get(name, {}))) for t in range(len(months))])
            dt = time.time() - t0
            ltag = tag if tag.lstrip("_") in ("behaviour", "comovement", "gravity") else ""   # метки с тегом — только для абляции слоёв
            np.save(OUT / f"labels_{name}_K{k}{ltag}.npy", L.astype(np.int32))
            timing.append({"method": name, "K": k, "seconds": dt, "layer": layer})
            for t, m in enumerate(months):
                rows.append({"month": m, "method": name, "K": k, "layer": layer, **evaluate.metric_panel(X[t], A_list[t], L[t])})
            print(f"{name:16s} K={k}: {dt:6.1f} c, медиана SW={np.median([r['SW'] for r in rows if r['method']==name and r['K']==k]):.3f}")
    long = pd.DataFrame(rows)
    long.to_csv(OUT / f"metrics_long{tag}.csv", index=False)
    pd.DataFrame(timing).to_csv(OUT / f"timing{tag}.csv", index=False)
    return long


def load_labels(name: str, k: int, tag: str = "") -> np.ndarray:
    return np.load(OUT / f"labels_{name}_K{k}{tag}.npy")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "temporal":      # отдельный процесс: только temporal Leiden, по одному K за раз
        mc = data.load_config("methods")
        for kk in mc["temporal_leiden"].get("k_grid", mc["k_grid"]):
            long = run(methods_subset=["temporal_leiden"], k_grid=[kk], tag=f"_temporal_K{kk}")
    elif len(sys.argv) > 1 and sys.argv[1] == "monthly":     # только помесячные методы
        long = run(methods_subset=list(data.load_config("methods")["methods"]))
    elif len(sys.argv) > 1 and sys.argv[1] == "only":        # один метод: python -m pulsar.run_cluster only <метод>
        long = run(methods_subset=[sys.argv[2]], tag=f"_{sys.argv[2]}")
    else:
        long = run()
    print(evaluate.summarize_over_months(long)[["SW", "CH", "S_Dbw", "AVI", "AVU", "MQ", "ANUI"]].round(3).to_string())
