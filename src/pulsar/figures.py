"""Статические рисунки для отчёта и слайдов → docs/figures/*.png. `python -m pulsar.figures`."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pulsar import data, features

OUTS = data.ROOT / "outputs"
FIG = data.ROOT / "docs" / "figures"
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999"]
RU = {"kmeans": "k-means", "ward": "Ward", "spectral": "Spectral", "leiden": "Leiden", "temporal_leiden": "temporal Leiden",
      "kefrin": "KEFRiN", "kefrin_balanced": "KEFRiN (сбаланс.)"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def _long() -> pd.DataFrame:
    skip = ("_behaviour", "_comovement", "_gravity")
    files = [f for f in sorted((OUTS / "cluster").glob("metrics_long*.csv")) if not any(f.stem.endswith(s) for s in skip)]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def fig_pareto(long: pd.DataFrame, k: int) -> None:
    sub = long[long.K == k].groupby("method")[["SW", "MQ", "ANUI"]].median()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for i, (m, r) in enumerate(sub.iterrows()):
        ax.scatter(r.SW, r.MQ, s=80, color=PALETTE[i % len(PALETTE)], zorder=3)
        ax.annotate(RU.get(m, m), (r.SW, r.MQ), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Silhouette (признаки) →"); ax.set_ylabel("Модулярность Q (сеть) →")
    ax.set_title(f"Фронт Парето: признаки и сеть, K = {k}", loc="left")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "pareto_sw_mq.png"); plt.close(fig)


def fig_k_curves(long: pd.DataFrame) -> None:
    metrics = [("SW", "Silhouette ↑"), ("MQ", "Модулярность Q ↑"), ("AVU", "AVU ↓"), ("ANUI", "ANUI ↑")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, (m, title) in zip(axes, metrics):
        for i, meth in enumerate(long.method.unique()):
            g = long[long.method == meth].groupby("K")[m].median()
            ax.plot(g.index, g.values, marker="o", ms=3, color=PALETTE[i % len(PALETTE)], label=RU.get(meth, meth))
        if m == "AVU":
            ks = np.arange(4, 11); ax.plot(ks, (ks - 1) / (2 * ks - 3), "k--", lw=1, label="случайный уровень")
        ax.set_title(title); ax.set_xlabel("K"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "k_curves.png"); plt.close(fig)


def fig_layers() -> None:
    s = pd.read_csv(OUTS / "graphs" / "network_summary.csv")
    cols = [("mean_degree", "средняя степень"), ("degree_assortativity", "ассортативность"), ("avg_clustering", "кластерный коэф."), ("rayleigh_quotient_norm", "негладкость признаков на графе")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (c, t) in zip(axes, cols):
        for i, layer in enumerate(["behaviour", "comovement", "gravity", "fused"]):
            g = s[s.layer == layer].set_index("month")[c]
            ax.plot(range(len(g)), g.values, color=PALETTE[i], label={"behaviour": "поведение", "comovement": "со-движение", "gravity": "география", "fused": "слитый"}[layer])
        ax.set_title(t); ax.set_xticks([0, 11, 23]); ax.set_xticklabels(["01.23", "12.23", "12.24"]); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "layers.png"); plt.close(fig)


def fig_dynamics(types: list[dict] | None) -> None:
    dyn = OUTS / "dynamics"
    names = {t["id"]: t["short"] for t in types} if types else {}
    sz = pd.read_csv(dyn / "sizes_over_time.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.stackplot(range(len(sz)), sz.T.values, labels=[names.get(int(c), f"тип {c}") for c in sz.columns], colors=PALETTE[: sz.shape[1]], alpha=0.9)
    ax.set_xticks([0, 11, 23]); ax.set_xticklabels(["01.2023", "12.2023", "12.2024"]); ax.set_title("Размеры подтверждённых типов по месяцам"); ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    tm = pd.read_csv(dyn / "transitions_2023-12_to_2024-12.csv")
    P = tm.pivot_table(index="from", columns="to", values="share_row", aggfunc="sum").fillna(0.0)
    ax = axes[1]
    im = ax.imshow(P.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(P.columns))); ax.set_yticks(range(len(P.index)))
    ax.set_xticklabels([names.get(int(c), c) for c in P.columns], rotation=45, ha="right", fontsize=8); ax.set_yticklabels([names.get(int(c), c) for c in P.index], fontsize=8)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            ax.text(j, i, f"{P.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if P.values[i, j] > 0.5 else "black")
    ax.set_title("Переходы 12.2023 → 12.2024 (подтверждённые типы)"); ax.set_xlabel("куда"); ax.set_ylabel("откуда")
    fig.tight_layout(); fig.savefig(FIG / "dynamics.png"); plt.close(fig)


def fig_marketplaces(types: list[dict] | None) -> None:
    f = features.load(); ids = f["ids"]; months = f["months"]
    mo = pd.read_parquet(OUTS / "interpret" / "mo_table.parquet").set_index("territory_id").reindex(ids)
    mode24 = mo["type_mode_2024"].astype(int).to_numpy()
    cons = data.read_consumption(); w = cons.pivot_table(index=["territory_id", "date"], columns="category", values="value")
    mp = (w["Маркетплейсы"] / w["Все категории"]).unstack("date").reindex(ids)
    names = {t["id"]: t["short"] for t in types} if types else {}
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in range(mode24.max() + 1):
        ax.plot(range(len(months)), mp[mode24 == k].median().values * 100, color=PALETTE[k], label=names.get(k, f"тип {k}"))
    ax.set_xticks([0, 11, 23]); ax.set_xticklabels(["01.2023", "12.2023", "12.2024"]); ax.set_ylabel("доля маркетплейсов в тратах, %")
    ax.set_title("Волна маркетплейсов по типам (медиана МО типа)"); ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "marketplaces.png"); plt.close(fig)


def fig_profiles(types: list[dict] | None) -> None:
    prof = pd.read_csv(OUTS / "interpret" / "mirkin_profile.csv")
    if "type" not in prof.columns or "feature" not in prof.columns:
        return
    piv = prof.pivot_table(index="type", columns="feature", values="mean")
    names = {t["id"]: t["short"] for t in types} if types else {}
    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([c.replace("clr_", "").replace("log_level_rel", "уровень") for c in piv.columns], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([names.get(int(i), i) for i in piv.index], fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:+.1f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="среднее признака в типе, σ")
    ax.set_title("Профили типов (стандартизованные признаки)")
    fig.tight_layout(); fig.savefig(FIG / "profiles.png"); plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    k = data.load_config("final")["k"]
    long = _long()
    fig_pareto(long, k); fig_k_curves(long); fig_layers()
    types = None
    tj = data.ROOT / "site" / "data" / "types.json"
    if tj.exists():
        types = json.loads(tj.read_text())
    for fn in (fig_dynamics, fig_marketplaces, fig_profiles):
        try:
            fn(types)
        except Exception as e:  # рисунки не должны ронять конвейер
            print(f"{fn.__name__}: пропущено ({e})")
    print(f"→ {FIG}")


if __name__ == "__main__":
    main()
