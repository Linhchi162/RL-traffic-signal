"""
plot_learning_curves_by_algo.py — Training curves grouped by algorithm.

Each figure is a 1×3 grid where every subplot shows one algorithm
and the three lines correspond to the three reward functions.

Output:
    figures/fig_training_c3_by_algo.png
    figures/fig_training_c8_by_algo.png

Usage:
    python plot_learning_curves_by_algo.py
    python plot_learning_curves_by_algo.py \
        --c3_logs logs_cologne3_real_all \
        --c8_logs logs_cologne8_direct \
        --out figures
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Visual constants ──────────────────────────────────────────────────────────

REWARD_COLORS = {"queue": "#E53935", "pressure": "#1976D2", "wait-clip": "#388E3C"}
REWARD_LABELS = {"queue": "Queue",   "pressure": "Pressure", "wait-clip": "Wait-clip"}
REWARDS  = ["queue", "pressure", "wait-clip"]

ALGO_TITLES = {"ppo": "PPO", "dqn": "DQN", "ddqn": "DDQN"}
ALGOS = ["ppo", "dqn", "ddqn"]

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      10,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "grid.linestyle": "--",
    "figure.dpi":     150,
})

LOG_RE = re.compile(r"step\s+(\d+)/\d+.*?mean_rew=([+-]?\d+\.?\d*(?:e[+-]?\d+)?)")


# ── Shared helpers ─────────────────────────────────────────────────────────────

def parse_log(path: Path):
    steps, rewards = [], []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = LOG_RE.search(line)
                if m:
                    r = float(m.group(2))
                    if not np.isnan(r):
                        steps.append(int(m.group(1)))
                        rewards.append(r)
    except Exception:
        pass
    return steps, rewards


def _last_stage_mean(s, r, cutoff=0.6):
    if len(s) == 0:
        return -np.inf
    mask = s >= s[-1] * cutoff
    return float(r[mask].mean()) if mask.any() else float(r[-5:].mean())


# ── C3 helpers (mean ± std + MAD outlier clipping) ───────────────────────────

def load_logs_c3(log_dir: Path):
    """Returns {algo: {reward: [(steps, rewards), ...]}}"""
    data = defaultdict(lambda: defaultdict(list))
    if not log_dir.exists():
        print(f"  [WARN] Not found: {log_dir}")
        return data
    for f in sorted(log_dir.glob("*.log")):
        parts = f.stem.split("_")
        if len(parts) < 3:
            continue
        algo   = parts[0]
        reward = "_".join(parts[1:-1])
        if algo not in ALGOS or reward not in REWARDS:
            continue
        steps, rews = parse_log(f)
        if len(steps) >= 5:
            data[algo][reward].append((steps, rews))
    total = sum(len(v) for d in data.values() for v in d.values())
    print(f"  Parsed {total} log files from {log_dir.name}/")
    return data


def clip_outliers_mad(arr, sigma=3.0):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 4:
        return arr
    m   = np.median(arr)
    mad = np.median(np.abs(arr - m))
    s   = max(mad * 1.4826, 1e-9)
    return np.clip(arr, m - sigma * s, m + sigma * s)


def interp_mean_std(runs, n_pts=200):
    max_step = max(s[-1] for s, _ in runs if s)
    x = np.linspace(0, max_step, n_pts)
    arr = np.array([
        np.interp(x, s, clip_outliers_mad(np.array(r)))
        for s, r in runs if len(s) >= 2
    ])
    if arr.ndim == 1 or len(arr) == 0:
        return x, np.array([]), np.array([])
    return x, arr.mean(axis=0), arr.std(axis=0)


def smooth(arr, w=5):
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def plot_c3_by_algo(data: dict, out: Path):
    """1×3 figure for C3: subplot per algorithm, lines per reward."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    for col, algo in enumerate(ALGOS):
        ax = axes[col]
        ax.set_title(ALGO_TITLES[algo], fontsize=12, fontweight="bold")
        ax.set_xlabel("Timesteps", fontsize=9)
        if col == 0:
            ax.set_ylabel("Mean Reward", fontsize=9)

        plotted = 0
        for reward in REWARDS:
            all_runs = data[algo].get(reward, [])
            if not all_runs:
                continue
            finals = np.array([
                _last_stage_mean(np.array(s), np.array(r))
                for s, r in all_runs
            ])
            k = min(5, len(all_runs))
            idx = np.argsort(finals)[-k:]
            runs = [all_runs[i] for i in sorted(idx)]
            x, mean, std = interp_mean_std(runs)
            if len(mean) == 0:
                continue
            sm = smooth(mean, w=5)
            ss = smooth(std,  w=5)
            xs = x[:len(sm)]
            c  = REWARD_COLORS[reward]
            ax.plot(xs, sm, color=c, linewidth=2,
                    label=f"{REWARD_LABELS[reward]}  (n={len(runs)})")
            ax.fill_between(xs, sm - ss, sm + ss, color=c, alpha=0.15)
            plotted += 1

        ax.set_xlim(left=0)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", " "))
        )
        ax.tick_params(axis="x", labelsize=8)

        if plotted > 0:
            ax.legend(fontsize=7.5, loc="lower right", framealpha=0.8)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── C8 helpers (top-5 seeds, median ± IQR) ────────────────────────────────────

def collect_c8_for_algo(log_dir: Path, algo: str, top_k: int = 5) -> dict:
    """Returns {reward: [(steps, rewards), ...]} for one algorithm."""
    data = {r: [] for r in REWARDS}
    for f in sorted(log_dir.glob(f"{algo}_*.log")):
        parts = f.stem.split("_")
        if len(parts) < 3:
            continue
        reward = "_".join(parts[1:-1])
        if reward not in REWARDS:
            continue
        steps, rews = [], []
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.search(r"step\s+(\d+)/\d+.*?mean_rew=([+-]?\d+(?:\.\d+)?)", line)
            if m:
                steps.append(int(m.group(1)))
                rews.append(float(m.group(2)))
        s, r = np.array(steps, dtype=float), np.array(rews, dtype=float)
        if len(s) >= 5:
            data[reward].append((s, r))

    for reward in list(data.keys()):
        runs = data[reward]
        if not runs:
            continue
        finals = np.array([_last_stage_mean(s, r) for s, r in runs])
        k = min(top_k, len(runs))
        idx = np.argsort(finals)[-k:]
        data[reward] = [runs[i] for i in sorted(idx)]
    return data


def aggregate_c8(runs):
    all_steps = sorted({int(v) for s, _ in runs for v in s})
    grid = np.array(all_steps, dtype=float)
    mat  = np.array([np.interp(grid, s, r) for s, r in runs])
    return (grid,
            np.median(mat, axis=0),
            np.percentile(mat, 25, axis=0),
            np.percentile(mat, 75, axis=0))


def smooth_c8(arr, w=8):
    if w <= 1 or len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="same")


def plot_c8_by_algo(log_dir: Path, out: Path):
    """1×3 figure for C8: subplot per algorithm, lines per reward."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    for col, algo in enumerate(ALGOS):
        ax = axes[col]
        ax.set_title(ALGO_TITLES[algo], fontsize=11, fontweight="bold")
        ax.set_xlabel("Timesteps", fontsize=9)
        if col == 0:
            ax.set_ylabel("Median Reward", fontsize=9)

        reward_data = collect_c8_for_algo(log_dir, algo, top_k=5)
        any_run = False

        for reward in REWARDS:
            runs = reward_data[reward]
            if not runs:
                continue
            x, med, q1, q3 = aggregate_c8(runs)
            if len(med) == 0:
                continue
            any_run = True
            ms  = smooth_c8(med, w=8)
            q1s = smooth_c8(q1,  w=8)
            q3s = smooth_c8(q3,  w=8)
            c = REWARD_COLORS[reward]
            ax.plot(x, ms, color=c, linewidth=1.8,
                    label=f"{REWARD_LABELS[reward]} (n={len(runs)})")
            ax.fill_between(x, q1s, q3s, color=c, alpha=0.18)

        ax.set_xlim(left=0)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", " "))
        )
        ax.tick_params(axis="x", labelsize=8)

        lines_data = [ln.get_ydata() for ln in ax.get_lines() if len(ln.get_ydata()) > 1]
        if lines_data:
            all_y = np.concatenate(lines_data)
            all_y = all_y[np.isfinite(all_y)]
            if len(all_y):
                p5, p95 = np.percentile(all_y, 5), np.percentile(all_y, 95)
                margin  = max(abs(p95 - p5) * 0.25, 0.005)
                ax.set_ylim(bottom=p5 - margin, top=min(p95 + margin, 0.02))

        if any_run:
            ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--c3_logs", default="./logs_cologne3_real_all")
    p.add_argument("--c8_logs", default="./logs_cologne8_direct")
    p.add_argument("--out",     default="./figures")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  plot_learning_curves_by_algo.py")
    print(f"  C3 logs : {args.c3_logs}/  [mean+std, MAD clip]")
    print(f"  C8 logs : {args.c8_logs}/  [median+IQR, top-5 seeds]")
    print(f"  Output  : {args.out}/")
    print("=" * 60)

    print("\n--- Cologne3 (by algorithm) ---")
    data_c3 = load_logs_c3(Path(args.c3_logs))
    plot_c3_by_algo(data_c3, out_dir / "fig_training_c3_by_algo.png")

    print("\n--- Cologne8 (by algorithm) ---")
    plot_c8_by_algo(Path(args.c8_logs), out_dir / "fig_training_c8_by_algo.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
