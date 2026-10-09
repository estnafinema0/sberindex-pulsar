"""Построение помесячных графов (три слоя + слияние) из признаков. `python -m pulsar.build_graphs`.

Выход: outputs/graphs/{layer}/{YYYY-MM}.npz и outputs/graphs/network_summary.csv (метрики сети по слоям и месяцам).
Граф месяца t использует только данные ≤ t: признаки месяца t (уже сглаженные причинно) и приросты за окно до t.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from pulsar import data, features, graphs

OUT = data.ROOT / "outputs" / "graphs"


def growth_window(growth: np.ndarray, t: int, window: int, min_window: int) -> np.ndarray | None:
    """Строки приростов за месяцы (t−window, t]; growth[j] — прирост от месяца j к j+1."""
    hi = t  # приросты до месяца t включительно: индексы 0..t-1
    lo = max(0, hi - window)
    G = growth[lo:hi]
    return G if G.shape[0] >= min_window else None


def build_all(cfg: dict | None = None, layers: tuple[str, ...] | None = None) -> pd.DataFrame:
    cfg = cfg or data.load_config("graph")
    f = features.load()
    X, growth, D, months = f["X"], f["growth"], f["D"].astype(np.float64), f["months"]
    rows = []
    for t, m in enumerate(months):
        G = growth_window(growth, t, cfg["comovement"]["window"], cfg["comovement"]["min_window"])
        t0 = time.time()
        built = graphs.build_month(X[t].astype(np.float64), G, D, cfg)
        for name, A in built.items():
            if layers and name not in layers:
                continue
            d = OUT / name
            d.mkdir(parents=True, exist_ok=True)
            graphs.save_graph(A, d / f"{m}.npz")
            s = graphs.network_summary(A, X[t].astype(np.float64))
            rows.append({"month": m, "layer": name, **s})
        print(f"{m}: {', '.join(f'{k}={v.nnz // 2}' for k, v in built.items())} рёбер, {time.time() - t0:.1f} c")
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "network_summary.csv", index=False)
    (OUT / "config_used.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    return summary


def load_graph(layer: str, month: str):
    return graphs.load_graph(OUT / layer / f"{month}.npz")


def load_layer(layer: str, months: list[str]) -> list:
    return [load_graph(layer, m) for m in months]


if __name__ == "__main__":
    s = build_all()
    print(s.groupby("layer")[["n_edges", "mean_degree", "n_components", "isolate_share", "degree_assortativity"]].median().round(3) if "n_edges" in s else s.head())
