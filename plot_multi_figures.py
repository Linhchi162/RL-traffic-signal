"""
plot_multi_figures.py — Sinh bieu do ket qua danh gia (phong cach paper RLTSCQ)

Dau vao (trong results/):
    summary.csv          — mean +- std theo (algo, reward, obs_mode, scope, flow)
    timeseries_results.csv — ket qua theo buoc mo phong

Dau ra:
    results/fig_main.png — 4-panel figure (a) line chart, (b/c/d) bar charts

Chay:
    python plot_multi_figures.py --results_dir ./results --flow_line high
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


# ------------------- tuning ------------------------------------------------
SMOOTH_WIN = 50     # buoc lam tron duong line chart

# Bieu do chi hien thi cac "nhom" nay (moi nhom = 1 duong / 1 thanh)
# Key = (algo, reward, obs_mode)  hoac  ("fixed",) hoac  ("webster",)
# Ten hien thi va mau sac
DISPLAY = {
    ("ppo",  "queue",   "raw"):      {"label": "PPO-19D (queue)",    "color": "#1f77b4", "ls": "-"},
    ("ppo",  "queue",   "baseline"): {"label": "PPO-7D (queue)",     "color": "#ff7f0e", "ls": "-"},
    ("ppo",  "delay",   "raw"):      {"label": "PPO-19D (delay)",    "color": "#2ca02c", "ls": "--"},
    ("ppo",  "pressure","raw"):      {"label": "PPO-19D (pressure)", "color": "#9467bd", "ls": "--"},
    ("ppo",  "speed",   "raw"):      {"label": "PPO-19D (speed)",    "color": "#8c564b", "ls": "--"},
    ("dqn",  "wait-clip","resco"):   {"label": "RESCO-DQN",         "color": "#d62728", "ls": "-"},
    ("webster", "-",    "-"):        {"label": "Dyn. Webster",       "color": "#17becf", "ls": "-."},
    ("fixed",   "-",    "-"):        {"label": "Fixed-time",         "color": "#7f7f7f", "ls": ":"},
}


def _smooth(arr, win):
    if win <= 1 or len(arr) < win:
        return np.array(arr, dtype=float)
    kernel = np.ones(win) / win
    return np.convolve(arr, kernel, mode="valid")


def _display_key(row):
    algo = row["algo"].lower()
    reward = row.get("reward", "-") or "-"
    obs = row.get("obs_mode", "-") or "-"
    if algo in ("fixed", "webster"):
        return (algo, "-", "-")
    return (algo, reward, obs)


def load_summary(results_dir: Path, flow: str, scope: str = "multi",
                 use_median: bool = True):
    """Doc summary.csv, tra ve {display_key: (center_queue, spread_queue)}.

    use_median=True : center=median_queue, spread=std_queue/2 (IQR proxy)
    use_median=False: center=mean_queue,   spread=std_queue
    """
    path = results_dir / "summary.csv"
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("scope", "multi") != scope:
                continue
            if row.get("flow", "-") not in (flow, "-", "paper"):
                continue
            dk = _display_key(row)
            if dk not in DISPLAY:
                continue
            try:
                if use_median and row.get("median_queue", ""):
                    mq = float(row["median_queue"])
                else:
                    mq = float(row["mean_queue"])
                sq = float(row.get("std_queue", 0))
            except ValueError:
                continue
            if dk not in out:
                out[dk] = (mq, sq)
            else:
                prev_mq, prev_sq = out[dk]
                out[dk] = ((prev_mq + mq) / 2, (prev_sq + sq) / 2)
    return out


def load_timeseries(results_dir: Path, flow: str):
    """Doc timeseries_results.csv, nhom theo display_key va tra ve {dk: list[float]}."""
    path = results_dir / "timeseries_results.csv"
    if not path.exists():
        return {}
    # Nhom cac buoc theo (dk, seed)
    by_key_seed = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("flow", "-") not in (flow, "-"):
                continue
            dk = _display_key(row)
            if dk not in DISPLAY:
                continue
            seed = row.get("seed", "0")
            try:
                q = float(row["total_queue"])
            except (ValueError, KeyError):
                continue
            by_key_seed[(dk, seed)].append(q)

    # Tinh mean theo seed, sau do tinh mean +- std qua cac seeds
    by_key: dict = defaultdict(list)
    for (dk, seed), vals in by_key_seed.items():
        by_key[dk].append(np.array(vals, dtype=float))

    result = {}
    for dk, seed_arrays in by_key.items():
        min_len = min(len(a) for a in seed_arrays)
        clipped = np.stack([a[:min_len] for a in seed_arrays], axis=0)
        mean_ts = clipped.mean(axis=0)
        std_ts  = clipped.std(axis=0)
        result[dk] = (mean_ts, std_ts)
    return result


def make_bar_ax(ax, summary: dict, title: str):
    """Ve bar chart cho 1 flow level."""
    keys_present = [dk for dk in DISPLAY if dk in summary]
    if not keys_present:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    means = [summary[dk][0] for dk in keys_present]
    stds  = [summary[dk][1] for dk in keys_present]
    labels = [DISPLAY[dk]["label"] for dk in keys_present]
    colors = [DISPLAY[dk]["color"] for dk in keys_present]

    x = np.arange(len(keys_present))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                  error_kw={"elinewidth": 1.2, "ecolor": "black"}, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean Queue Length (veh)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


def make_line_ax(ax, ts_data: dict, flow_label: str):
    """Ve line chart (mean +- std band)."""
    if not ts_data:
        ax.set_title(f"Queue vs Step ({flow_label} flow)")
        ax.text(0.5, 0.5, "No timeseries data", ha="center", va="center",
                transform=ax.transAxes)
        return

    for dk, (mean_ts, std_ts) in ts_data.items():
        cfg = DISPLAY[dk]
        s_mean = _smooth(mean_ts, SMOOTH_WIN)
        s_std  = _smooth(std_ts,  SMOOTH_WIN)
        xs = np.arange(len(s_mean))
        ax.plot(xs, s_mean, label=cfg["label"], color=cfg["color"],
                linestyle=cfg["ls"], linewidth=1.5)
        ax.fill_between(xs, s_mean - s_std, s_mean + s_std,
                        color=cfg["color"], alpha=0.12)

    ax.set_xlabel("Simulation Step", fontsize=9)
    ax.set_ylabel("Total Queue Length (veh)", fontsize=9)
    ax.set_title(f"(a) Queue vs Step — {flow_label.capitalize()} Flow",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)


def parse_args():
    p = argparse.ArgumentParser(description="Sinh 4-panel figure tu ket qua danh gia.")
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--scope", default="multi", choices=["single", "multi"])
    p.add_argument("--flow_line", default="high",
                   choices=["high", "medium", "low"],
                   help="Flow level cho line chart panel (a)")
    p.add_argument("--out", default=None,
                   help="Duong dan luu PNG (default: results_dir/fig_main.png)")
    p.add_argument("--use_mean", action="store_true",
                   help="Dung mean thay vi median cho bar charts (default: median)")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_path = Path(args.out) if args.out else results_dir / "fig_main.png"
    use_median = not args.use_mean

    flows = ["high", "medium", "low"] if args.scope == "multi" else ["paper"]
    flow_labels = {"high": "High", "medium": "Medium", "low": "Low", "paper": "Paper"}

    center_label = "Median" if use_median else "Mean"

    # Load data
    ts_data = load_timeseries(results_dir, args.flow_line)
    summary_high   = load_summary(results_dir, flows[0], args.scope, use_median)
    summary_medium = load_summary(results_dir, flows[1] if len(flows) > 1 else flows[0], args.scope, use_median)
    summary_low    = load_summary(results_dir, flows[2] if len(flows) > 2 else flows[0], args.scope, use_median)

    # --- Build figure ---
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # (a) Line chart — top row spanning all 3 columns
    ax_line = fig.add_subplot(gs[0, :])
    make_line_ax(ax_line, ts_data, flow_labels.get(args.flow_line, args.flow_line))

    # (b) (c) (d) bar charts — bottom row
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    fl0 = flow_labels.get(flows[0], flows[0])
    fl1 = flow_labels.get(flows[1] if len(flows) > 1 else flows[0],
                          flows[1] if len(flows) > 1 else flows[0])
    fl2 = flow_labels.get(flows[2] if len(flows) > 2 else flows[0],
                          flows[2] if len(flows) > 2 else flows[0])

    make_bar_ax(ax_b, summary_high,   f"(b) {fl0} Flow")
    make_bar_ax(ax_c, summary_medium, f"(c) {fl1} Flow")
    make_bar_ax(ax_d, summary_low,    f"(d) {fl2} Flow")

    fig.suptitle(f"Traffic Signal Control — Multi-Intersection Grid 2×2\n"
                 f"{center_label} Queue Length Comparison", fontsize=13, fontweight="bold", y=1.01)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Also save individual bar chart with all flows side by side (alternative view)
    _save_grouped_bar(results_dir, args.scope, flows, flow_labels, out_path.parent)


def _save_grouped_bar(results_dir, scope, flows, flow_labels, out_dir):
    """Luu them 1 bieu do bar chart nhom theo flow (tuong tu Fig 5b/c/d paper)."""
    all_data = {}
    for flow in flows:
        all_data[flow] = load_summary(results_dir, flow, scope)

    keys_present = [dk for dk in DISPLAY
                    if any(dk in all_data[f] for f in flows)]
    if not keys_present:
        return

    n_groups = len(keys_present)
    n_flows  = len(flows)
    x = np.arange(n_groups)
    width = 0.8 / n_flows

    cmap_flow = {"high": "#1f77b4", "medium": "#ff7f0e", "low": "#2ca02c", "paper": "#9467bd"}

    fig, ax = plt.subplots(figsize=(max(9, n_groups * 1.4), 5))
    for fi, flow in enumerate(flows):
        sd = all_data[flow]
        means = [sd.get(dk, (0, 0))[0] for dk in keys_present]
        stds  = [sd.get(dk, (0, 0))[1] for dk in keys_present]
        offset = (fi - n_flows / 2 + 0.5) * width
        ax.bar(x + offset, means, width * 0.9, yerr=stds, capsize=3,
               label=f"{flow_labels.get(flow, flow)} Flow",
               color=cmap_flow.get(flow, f"C{fi}"),
               error_kw={"elinewidth": 1, "ecolor": "black"}, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[dk]["label"] for dk in keys_present],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Queue Length (veh)", fontsize=10)
    ax.set_title("Mean Queue Length by Method and Flow Level", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    out = out_dir / "fig_bar_grouped.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
