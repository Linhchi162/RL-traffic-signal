"""
plot_paper_figures.py
Vẽ 4 biểu đồ theo đúng layout trong bài báo:
  (a) Total Queue Length vs Cycle Number
  (b) Queue Length vs Phase - High Flow
  (c) Queue Length vs Phase - Medium Flow
  (d) Queue Length vs Phase - Low Flow
"""

import math
import os
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless - khong can cua so GUI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 150,
})

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots_paper"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CYCLE_CSV = RESULTS_DIR / "paper_cycle_results.csv"
PHASE_CSV = RESULTS_DIR / "paper_phase_results.csv"

# ---------------------------------------------------------------------------
# Tên hiển thị & màu sắc — khớp với bảng bài báo
# ---------------------------------------------------------------------------
LABEL_MAP = {
    ("ppo", "queue"):           "PPO (Queue)",
    ("ppo", "diff-waiting-time"): "PPO (Delay)",
    ("ppo", "average-speed"):   "PPO (Speed)",
    ("ppo", "pressure"):        "PPO (Pressure)",
    ("ppo", "queue-AE-4d"):     "PPO (Queue AE-4D)",
    ("ppo", "queue-AE-8d"):     "PPO (Queue AE-8D)",
    ("ppo", "queue-AE-16d"):    "PPO (Queue AE-16D)",
    ("ppo", "queue-AE-19d"):    "PPO (Queue AE-19D)",
    ("ppo", "queue-AE-32d"):    "PPO (Queue AE-32D)",
    ("dqn", "wait-clip"):       "RESCO-DQN",
    ("webster", "-"):           "Dynamic Webster",
    ("fixed", "-"):             "Fixed-time",
}

PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#a6d854", "#66c2a5", "#fc8d62", "#8da0cb",
]

def build_color_map(labels):
    return {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(labels)}


# ---------------------------------------------------------------------------
# Phần (a): Total Queue Length vs Cycle Number
# ---------------------------------------------------------------------------
def plot_queue_vs_cycle(ax, df):
    groups = []
    for (algo, reward), g in df.groupby(["algo", "reward"]):
        label = LABEL_MAP.get((algo, reward), f"{algo}/{reward}")
        groups.append((label, g))

    color_map = build_color_map([l for l, _ in groups])

    # Align cycles by taking common range
    max_cycle = df["cycle_num"].max()
    cycles = np.arange(1, max_cycle + 1)

    for label, g in groups:
        color = color_map[label]
        seeds = g["seed"].unique()
        seed_series = []
        for seed in seeds:
            sg = g[g["seed"] == seed].sort_values("cycle_num")
            if len(sg) < 2:
                continue
            # Reindex to common cycle range
            s = sg.set_index("cycle_num")["total_queue"].reindex(cycles)
            seed_series.append(s.values)

        if not seed_series:
            # Baselines (no seed column) — single trajectory
            sg = g.sort_values("cycle_num")
            s = sg.set_index("cycle_num")["total_queue"].reindex(cycles).values
            ax.plot(cycles, s, color=color, linewidth=1.2, label=label)
            continue

        arr = np.array(seed_series, dtype=float)
        mean = np.nanmean(arr, axis=0)
        std  = np.nanstd(arr, axis=0)

        valid = ~np.isnan(mean)
        ax.plot(cycles[valid], mean[valid], color=color, linewidth=1.2, label=label)
        ax.fill_between(cycles[valid],
                        mean[valid] - std[valid],
                        mean[valid] + std[valid],
                        alpha=0.15, color=color)

    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("Total Queue Length (vehicles)")
    ax.set_title("(a) Total Queue Length vs Cycle Number")
    ax.set_xlim(cycles[0], cycles[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", ncol=2, framealpha=0.7)


# ---------------------------------------------------------------------------
# Phần (b/c/d): Queue Length vs Phase per Flow
# ---------------------------------------------------------------------------
PHASES = ["EW Through + Left", "EW Right", "NS Through + Left", "NS Right"]

def plot_queue_vs_phase(ax, df_phase, flow, title_suffix):
    df = df_phase[df_phase["flow"] == flow].copy()

    # Aggregate over seeds
    agg_rows = []
    for (algo, reward), g in df.groupby(["algo", "reward"]):
        label = LABEL_MAP.get((algo, reward), f"{algo}/{reward}")
        means = {}
        stds  = {}
        for ph in PHASES:
            if ph in g.columns:
                vals = g[ph].dropna().values.astype(float)
                means[ph] = vals.mean() if len(vals) > 0 else np.nan
                stds[ph]  = vals.std()  if len(vals) > 1 else 0.0
            else:
                means[ph] = np.nan
                stds[ph]  = 0.0
        agg_rows.append({"label": label, "means": means, "stds": stds})

    if not agg_rows:
        ax.set_visible(False)
        return

    labels     = [r["label"] for r in agg_rows]
    color_map  = build_color_map(labels)

    n_algos  = len(agg_rows)
    n_phases = len(PHASES)
    group_w  = 0.8
    bar_w    = group_w / n_algos
    x_base   = np.arange(n_phases)

    for i, row in enumerate(agg_rows):
        offsets = x_base + (i - n_algos / 2 + 0.5) * bar_w
        vals  = [row["means"][ph] for ph in PHASES]
        errs  = [row["stds"][ph]  for ph in PHASES]
        color = color_map[row["label"]]
        ax.bar(offsets, vals, bar_w * 0.9, yerr=errs, color=color,
               capsize=2, label=row["label"], error_kw={"linewidth": 0.7})

    ax.set_xlabel("Phase")
    ax.set_ylabel("Average Queue Length (vehicles)")
    ax.set_title(f"({title_suffix}) Queue Length vs Phase - {flow.capitalize()} Flow Conditions")
    ax.set_xticks(x_base)
    ax.set_xticklabels(PHASES, fontsize=7)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def filter_diverged(df_phase):
    """Loại bỏ các hàng mà bất kỳ phase nào có queue > ngưỡng (model bị diverge)."""
    DIVERGE_THRESHOLD = 20  # vehicles/phase
    phase_cols = [c for c in PHASES if c in df_phase.columns]
    max_queue = df_phase[phase_cols].max(axis=1)
    n_before = len(df_phase)
    df_clean = df_phase[max_queue <= DIVERGE_THRESHOLD].copy()
    removed = n_before - len(df_clean)
    if removed > 0:
        bad = df_phase[max_queue > DIVERGE_THRESHOLD][["algo","reward","seed","flow"]].drop_duplicates()
        print(f"[FILTER] Loai bo {removed} hang bi diverge (queue > {DIVERGE_THRESHOLD}):")
        print(bad.to_string(index=False))
    return df_clean


def filter_diverged_cycles(df_cycle, df_phase):
    """Loại bỏ các dòng cycle tương ứng với các seed+reward bị diverge trong phase data."""
    DIVERGE_THRESHOLD = 20
    phase_cols = [c for c in PHASES if c in df_phase.columns]
    bad_keys = df_phase[df_phase[phase_cols].max(axis=1) > DIVERGE_THRESHOLD][
        ["algo", "reward", "seed", "flow"]
    ].drop_duplicates()

    if bad_keys.empty:
        return df_cycle

    mask = pd.Series([False] * len(df_cycle), index=df_cycle.index)
    for _, row in bad_keys.iterrows():
        m = (
            (df_cycle["algo"] == row["algo"]) &
            (df_cycle["reward"] == row["reward"]) &
            (df_cycle["seed"] == str(row["seed"])) &
            (df_cycle["flow"] == row["flow"])
        )
        mask |= m
    return df_cycle[~mask].copy()


def main():
    if not CYCLE_CSV.exists():
        print(f"[ERROR] Khong tim thay file: {CYCLE_CSV}")
        return
    if not PHASE_CSV.exists():
        print(f"[ERROR] Khong tim thay file: {PHASE_CSV}")
        return

    df_cycle = pd.read_csv(CYCLE_CSV)
    df_phase = pd.read_csv(PHASE_CSV)

    # Normalise seed column for baselines
    if "seed" not in df_cycle.columns:
        df_cycle["seed"] = "-"
    df_cycle["seed"] = df_cycle["seed"].fillna("-").astype(str)

    if "seed" not in df_phase.columns:
        df_phase["seed"] = "-"
    df_phase["seed"] = df_phase["seed"].fillna("-").astype(str)

    # ==== LOC BO CAC MODEL BI DIVERGE ====
    df_phase   = filter_diverged(df_phase)
    df_cycle   = filter_diverged_cycles(df_cycle, pd.read_csv(PHASE_CSV))

    # high-flow only for cycle plot (matches paper Fig 5a)
    df_cycle_high = df_cycle[df_cycle["flow"] == "high"]

    # ---- Build shared legend from filtered data ----
    all_labels = list(dict.fromkeys(
        LABEL_MAP.get((r.algo, r.reward), f"{r.algo}/{r.reward}")
        for r in df_cycle.itertuples()
    ))
    color_map = build_color_map(all_labels)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Traffic Signal Control — Performance Analysis (Filtered)", fontsize=12, y=0.98)

    # (a) Cycle plot
    plot_queue_vs_cycle(axes[0, 0], df_cycle_high)

    # (b), (c), (d)
    for ax, flow, suffix in [
        (axes[0, 1], "high",   "b"),
        (axes[1, 0], "medium", "c"),
        (axes[1, 1], "low",    "d"),
    ]:
        plot_queue_vs_phase(ax, df_phase, flow, suffix)

    # Global legend below figure
    handles = [
        mpatches.Patch(color=color_map.get(lbl, "#aaaaaa"), label=lbl)
        for lbl in all_labels
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(all_labels), 6),
        bbox_to_anchor=(0.5, -0.02),
        framealpha=0.9,
        fontsize=7,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    out_path = PLOTS_DIR / "figure5_paper.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Da luu bien do: {out_path}")
    # plt.show()  # disabled in headless mode


if __name__ == "__main__":
    main()
