"""
plot_multi_figures.py — Sinh bieu do ket qua danh gia

Su dung:
    python plot_multi_figures.py --results_dir ./results_multi --scope multi
    python plot_multi_figures.py --results_dir ./results_multi --boxplot
    python plot_multi_figures.py --results_dir ./results_multi --scope multi --flow_line high
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SMOOTH_WIN = 50

# ---------------------------------------------------------------------------
# Algorithm display config
# (algo, reward, obs_mode) -> {label, color, ls, group_color}
# ---------------------------------------------------------------------------
DISPLAY_MULTI = {
    ("fixed",   "-",         "-"):        {"label": "Fixed-time",        "color": "#7f7f7f", "ls": ":"},
    ("webster", "-",         "-"):        {"label": "Webster",           "color": "#17becf", "ls": "-."},
    ("dqn",     "queue",     "resco"):    {"label": "DQN-Queue",         "color": "#1f77b4", "ls": "-"},
    ("dqn",     "pressure",  "resco"):    {"label": "DQN-Pressure",      "color": "#4a90d9", "ls": "--"},
    ("dqn",     "wait-clip", "resco"):    {"label": "DQN-WaitClip",      "color": "#7ab3e8", "ls": ":"},
    ("ddqn",    "queue",     "resco"):    {"label": "DDQN-Queue",        "color": "#d62728", "ls": "-"},
    ("ddqn",    "pressure",  "resco"):    {"label": "DDQN-Pressure",     "color": "#e87878", "ls": "--"},
    ("ddqn",    "wait-clip", "resco"):    {"label": "DDQN-WaitClip★",   "color": "#f5b8b8", "ls": ":"},
    ("ppo",     "queue",     "baseline"): {"label": "PPO-7D-Queue",      "color": "#2ca02c", "ls": "-"},
    ("ppo",     "pressure",  "baseline"): {"label": "PPO-7D-Pressure",   "color": "#6bca6b", "ls": "--"},
    ("ppo",     "wait-clip", "baseline"): {"label": "PPO-7D-WaitClip",   "color": "#b3e8b3", "ls": ":"},
}

DISPLAY_SINGLE = DISPLAY_MULTI

FLOW_LABELS = {"high": "High Flow", "medium": "Medium Flow", "low": "Low Flow", "paper": "Paper Flow"}


def _smooth(arr, win):
    if win <= 1 or len(arr) < win:
        return np.array(arr, dtype=float)
    return np.convolve(arr, np.ones(win) / win, mode="valid")


def _display_key(row, display=None):
    algo = row["algo"].lower()
    reward = row.get("reward", "-") or "-"
    obs = row.get("obs_mode", "-") or "-"
    if algo in ("fixed", "webster"):
        return (algo, "-", "-")
    return (algo, reward, obs)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_summary(results_dir, flow, scope, display, use_median=True):
    path = results_dir / "summary.csv"
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("scope", "multi") != scope:
                continue
            if row.get("flow", "-") not in (flow, "-", "paper"):
                continue
            dk = _display_key(row, display)
            if dk not in display:
                continue
            try:
                mq = float(row["median_queue"] if use_median and row.get("median_queue") else row["mean_queue"])
                sq = float(row.get("std_queue", 0))
            except ValueError:
                continue
            if dk not in out:
                out[dk] = (mq, sq)
    return out


def load_timeseries(results_dir, flow, display):
    path = results_dir / "timeseries_results.csv"
    if not path.exists():
        return {}
    by_key_seed = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("flow", "-") not in (flow, "-"):
                continue
            dk = _display_key(row, display)
            if dk not in display:
                continue
            seed = row.get("seed", "0")
            try:
                q = float(row["total_queue"])
            except (ValueError, KeyError):
                continue
            by_key_seed[(dk, seed)].append(q)

    by_key = defaultdict(list)
    for (dk, seed), vals in by_key_seed.items():
        by_key[dk].append(np.array(vals, dtype=float))

    result = {}
    for dk, arrays in by_key.items():
        min_len = min(len(a) for a in arrays)
        clipped = np.stack([a[:min_len] for a in arrays])
        result[dk] = (clipped.mean(0), clipped.std(0))
    return result


def load_boxplot_data(results_dir, metric, flow, scope="multi"):
    """Load per-seed values for box plotting."""
    path = results_dir / "all_results.csv"
    if not path.exists():
        return {}
    data = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("scope", "multi") != scope:
                continue
            if flow != "all" and row.get("flow", "-") not in (flow, "-", "paper"):
                continue
            dk = _display_key(row)
            try:
                val = float(row[metric])
            except (ValueError, KeyError):
                continue
            data[dk].append(val)
    return data


# ---------------------------------------------------------------------------
# Axes builders
# ---------------------------------------------------------------------------

def make_bar_ax(ax, summary, display, title, ylabel=True, metric_label="Queue Length (veh)"):
    keys_present = [dk for dk in display if dk in summary]
    if not keys_present:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    means  = [summary[dk][0] for dk in keys_present]
    stds   = [summary[dk][1] for dk in keys_present]
    labels = [display[dk]["label"] for dk in keys_present]
    colors = [display[dk]["color"] for dk in keys_present]
    x = np.arange(len(keys_present))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors,
           error_kw={"elinewidth": 1.2, "ecolor": "black"}, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    if ylabel:
        ax.set_ylabel(metric_label, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


def make_line_ax(ax, ts_data, display, title):
    if not ts_data:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No timeseries data", ha="center", va="center", transform=ax.transAxes)
        return
    for dk, (mean_ts, std_ts) in ts_data.items():
        cfg = display[dk]
        s_mean = _smooth(mean_ts, SMOOTH_WIN)
        s_std  = _smooth(std_ts,  SMOOTH_WIN)
        xs = np.arange(len(s_mean))
        ax.plot(xs, s_mean, label=cfg["label"], color=cfg["color"],
                linestyle=cfg["ls"], linewidth=1.5)
        ax.fill_between(xs, s_mean - s_std, s_mean + s_std, color=cfg["color"], alpha=0.12)
    ax.set_xlabel("Simulation Step", fontsize=9)
    ax.set_ylabel("Total Queue Length (veh)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)


def make_boxplot_ax(ax, box_data, display, title, ylabel=True, metric_label="Queue Length (veh)"):
    """Box plot: one box per algorithm config, variation across seeds."""
    keys_present = [dk for dk in display if dk in box_data and box_data[dk]]
    if not keys_present:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    plot_data  = [box_data[dk] for dk in keys_present]
    colors     = [display[dk]["color"] for dk in keys_present]
    labels     = [display[dk]["label"] for dk in keys_present]

    bp = ax.boxplot(plot_data, patch_artist=True, widths=0.6,
                    medianprops={"color": "black", "linewidth": 2},
                    whiskerprops={"linewidth": 1.2},
                    capprops={"linewidth": 1.2},
                    flierprops={"marker": "o", "markersize": 4, "alpha": 0.6})

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(keys_present) + 1))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    if ylabel:
        ax.set_ylabel(metric_label, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def plot_boxplots(results_dir, out_dir, display, scope="multi"):
    """Fig: Box plots for queue (top row) and wait per vehicle (bottom row), 3 flow conditions."""
    flows = ["high", "medium", "low"] if scope == "multi" else ["paper"]
    ncols = len(flows)

    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 10))
    if ncols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle("Distribution of Traffic Metrics across Seeds\n(Box = median + IQR, whiskers = min/max)",
                 fontsize=13, fontweight="bold")

    for col, flow in enumerate(flows):
        q_data = load_boxplot_data(results_dir, "mean_queue", flow, scope)
        w_data = load_boxplot_data(results_dir, "mean_wait_per_veh", flow, scope)
        lbl = FLOW_LABELS.get(flow, flow)
        make_boxplot_ax(axes[0, col], q_data, display, f"Avg Queue — {lbl}",
                        ylabel=(col == 0), metric_label="Queue Length (veh/intersection)")
        make_boxplot_ax(axes[1, col], w_data, display, f"Avg Wait/Vehicle — {lbl}",
                        ylabel=(col == 0), metric_label="Wait Time (s/vehicle)")

    plt.tight_layout()
    out = out_dir / "fig_boxplot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_timeseries_multi(results_dir, out_dir, display, flow_line="high"):
    """Fig: Line chart with ±1 std shaded region across seeds."""
    ts_data = load_timeseries(results_dir, flow_line, display)

    fig, ax = plt.subplots(figsize=(14, 5))
    make_line_ax(ax, ts_data, display,
                 f"Queue vs Simulation Step — {FLOW_LABELS.get(flow_line, flow_line)}\n(mean ± 1 std across seeds)")
    plt.tight_layout()
    out = out_dir / f"fig_timeseries_{flow_line}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_bar_flows(results_dir, out_dir, display, scope="multi", use_median=True):
    """Fig: Bar chart — average queue per flow condition (High / Medium / Low)."""
    flows = ["high", "medium", "low"] if scope == "multi" else ["paper"]
    center = "Median" if use_median else "Mean"

    fig, axes = plt.subplots(1, len(flows), figsize=(5 * len(flows), 6), sharey=True)
    if len(flows) == 1:
        axes = [axes]

    fig.suptitle(f"Average Queue Length by Flow Condition ({center} ± Std, across seeds)",
                 fontsize=13, fontweight="bold")

    for ax, flow in zip(axes, flows):
        summary = load_summary(results_dir, flow, scope, display, use_median)
        make_bar_ax(ax, summary, display, FLOW_LABELS.get(flow, flow),
                    ylabel=(flow == flows[0]), metric_label="Queue Length (veh/intersection)")

    plt.tight_layout()
    out = out_dir / "fig_bar_flows.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_throughput_travel(results_dir, out_dir, display, scope="multi"):
    """Fig: Throughput and Mean Travel Time bar charts."""
    flows = ["high", "medium", "low"] if scope == "multi" else ["paper"]
    path = results_dir / "summary.csv"
    if not path.exists():
        print("[WARN] summary.csv not found, skip throughput/travel time plot")
        return

    def load_metric(metric):
        out = {}
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("scope", "multi") != scope:
                    continue
                dk = _display_key(row)
                if dk not in display:
                    continue
                flow = row.get("flow", "-")
                try:
                    val = float(row.get(metric, 0) or 0)
                    std = float(row.get(f"std_{metric.replace('mean_','')}", 0) or 0)
                except ValueError:
                    continue
                out[(dk, flow)] = (val, std)
        return out

    thru_data  = load_metric("mean_throughput")
    ttime_data = load_metric("mean_travel_time")

    fig, axes = plt.subplots(2, len(flows), figsize=(5 * len(flows), 10))
    if len(flows) == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle("Throughput & Mean Travel Time by Flow Condition", fontsize=13, fontweight="bold")

    for col, flow in enumerate(flows):
        lbl = FLOW_LABELS.get(flow, flow)
        for row_idx, (data, ylabel, title_prefix) in enumerate([
            (thru_data,  "Vehicles completed",    "Throughput"),
            (ttime_data, "Travel Time (s/vehicle)", "Mean Travel Time"),
        ]):
            ax = axes[row_idx, col]
            keys_present = [dk for dk in display if (dk, flow) in data]
            if not keys_present:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{title_prefix} — {lbl}")
                continue
            means  = [data[(dk, flow)][0] for dk in keys_present]
            stds   = [data[(dk, flow)][1] for dk in keys_present]
            labels = [display[dk]["label"] for dk in keys_present]
            colors = [display[dk]["color"] for dk in keys_present]
            x = np.arange(len(keys_present))
            ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                   error_kw={"elinewidth": 1.2, "ecolor": "black"}, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f"{title_prefix} — {lbl}", fontsize=10, fontweight="bold")
            ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
            ax.set_axisbelow(True)

    plt.tight_layout()
    out = out_dir / "fig_throughput_travel.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_single(results_dir, out_path, use_median):
    display = DISPLAY_SINGLE
    center = "Median" if use_median else "Mean"
    summary = load_summary(results_dir, "paper", "single", display, use_median)
    ts_data = load_timeseries(results_dir, "paper", display)

    fig, (ax_line, ax_bar) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Traffic Signal Control — Single Intersection\n{center} Queue Length",
                 fontsize=13, fontweight="bold")
    make_line_ax(ax_line, ts_data, display, "(a) Queue vs Simulation Step")
    make_bar_ax(ax_bar, summary, display, f"(b) {center} Queue — Paper Flow")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_multi(results_dir, out_path, flow_line, use_median):
    display = DISPLAY_MULTI
    center = "Median" if use_median else "Mean"

    ts_data      = load_timeseries(results_dir, flow_line, display)
    summary_high = load_summary(results_dir, "high",   "multi", display, use_median)
    summary_med  = load_summary(results_dir, "medium", "multi", display, use_median)
    summary_low  = load_summary(results_dir, "low",    "multi", display, use_median)

    fig = plt.figure(figsize=(15, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.35)

    ax_line = fig.add_subplot(gs[0, :])
    make_line_ax(ax_line, ts_data, display,
                 f"(a) Queue vs Step — {FLOW_LABELS.get(flow_line, flow_line)}\n(mean ± 1 std across seeds)")

    make_bar_ax(fig.add_subplot(gs[1, 0]), summary_high, display, "(b) High Flow")
    make_bar_ax(fig.add_subplot(gs[1, 1]), summary_med,  display, "(c) Medium Flow", ylabel=False)
    make_bar_ax(fig.add_subplot(gs[1, 2]), summary_low,  display, "(d) Low Flow",    ylabel=False)

    fig.suptitle(f"Traffic Signal Control — Multi-Intersection Grid 2×2\n"
                 f"{center} Queue Length Comparison", fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--scope", default="multi", choices=["single", "multi"])
    p.add_argument("--flow_line", default="high", choices=["high", "medium", "low"],
                   help="Flow level cho line chart")
    p.add_argument("--out", default=None)
    p.add_argument("--use_mean", action="store_true")
    p.add_argument("--boxplot", action="store_true",
                   help="Sinh box plot (fig_boxplot.png)")
    p.add_argument("--all", action="store_true",
                   help="Sinh tat ca cac bieu do (main + boxplot + timeseries + throughput)")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    use_median  = not args.use_mean
    display     = DISPLAY_MULTI if args.scope == "multi" else DISPLAY_SINGLE
    out_path    = Path(args.out) if args.out else results_dir / (
        "fig_single.png" if args.scope == "single" else "fig_main.png")

    if args.all or args.boxplot:
        plot_boxplots(results_dir, results_dir, display, scope=args.scope)

    if args.all:
        plot_timeseries_multi(results_dir, results_dir, display, flow_line=args.flow_line)
        plot_bar_flows(results_dir, results_dir, display, scope=args.scope, use_median=use_median)
        plot_throughput_travel(results_dir, results_dir, display, scope=args.scope)

    if not args.boxplot or args.all:
        if args.scope == "single":
            plot_single(results_dir, out_path, use_median)
        else:
            plot_multi(results_dir, out_path, args.flow_line, use_median)


if __name__ == "__main__":
    main()
