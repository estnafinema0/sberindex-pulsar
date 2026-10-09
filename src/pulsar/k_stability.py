"""Устойчивость разбиения по K: бутстрэп по подвыборкам МО (Hennig 2007 / Ben-Hur 2002). `python -m pulsar.k_stability`.

Для каждого K и метода: n_boot подвыборок по frac МО → кластеризация → ARI между парами подвыборок на общих МО
(медиана и IQR), а также средний Жаккар кластеров с полным разбиением после венгерского сопоставления.
Выход: outputs/compare/k_stability.csv. Используется в правиле выбора K вместе с z-оценками индексов.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from pulsar import align, build_graphs, data, features, methods

OUT = data.ROOT / "outputs" / "compare"


def stability_for(X, A, k: int, method: str, params: dict, n_boot: int, frac: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    full = methods.run_method(method, X, A, k, seed=seed, params=dict(params))
    subs, labs = [], []
    for b in range(n_boot):
        idx = np.sort(rng.choice(N, size=int(frac * N), replace=False))
        lab = methods.run_method(method, X[idx], A[idx][:, idx], k, seed=seed + b + 1, params=dict(params))
        subs.append(idx); labs.append(lab)
    aris = []
    for (ia, la), (ib, lb) in itertools.combinations(zip(subs, labs), 2):
        common, pa, pb = np.intersect1d(ia, ib, return_indices=True)
        aris.append(adjusted_rand_score(la[pa], lb[pb]))
    # Жаккар кластеров с полным разбиением (устойчивость отдельных кластеров)
    jac = []
    for idx, lab in zip(subs, labs):
        J = align.jaccard_matrix(full[idx], lab)
        jac.append(J.max(axis=1))  # для каждого кластера полного разбиения — лучший Жаккар в подвыборке
    jac = np.vstack([j if len(j) == k else np.pad(j, (0, k - len(j))) for j in jac])
    return {"method": method, "K": k, "ari_median": float(np.median(aris)), "ari_iqr": float(np.subtract(*np.percentile(aris, [75, 25]))),
            "jaccard_mean": float(jac.mean()), "jaccard_min_cluster": float(jac.mean(axis=0).min()), "n_boot": n_boot}


def main(month: str = "2024-06", methods_list=("kmeans", "kefrin_balanced", "leiden"), n_boot: int = 10, frac: float = 0.8) -> pd.DataFrame:
    mcfg = data.load_config("methods")
    f = features.load(); t = f["months"].index(month)
    X = f["X"][t].astype(float); A = build_graphs.load_graph(data.load_config("graph")["primary"], month)
    rows = []
    for method in methods_list:
        for k in mcfg["k_grid"]:
            r = stability_for(X, A, k, method, mcfg["params"].get(method, {}), n_boot, frac, mcfg["seed"])
            r["month"] = month; rows.append(r)
            print(f"{method:16s} K={k}: ARI {r['ari_median']:.3f} (IQR {r['ari_iqr']:.3f}), Жаккар {r['jaccard_mean']:.3f}, мин. кластер {r['jaccard_min_cluster']:.3f}")
    df = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True); df.to_csv(OUT / "k_stability.csv", index=False)
    return df


if __name__ == "__main__":
    main()
