"""
plot_learning_curves.py — Training reward curves.

Creates TWO separate 1x3 figures using the exact methods that produced
the clean reference plots:

  fig_training_c3.png — Cologne3 (real route, no curriculum)
    Method: mean +- std with MAD outlier clipping  [from plot_training_curves.py]
    Data  : logs_cologne3_real_all  (10 seeds)

  fig_training_c8.png — Cologne8 (direct training, random route)
    Method: median +- IQR                          [from original plot_learning_curves.py]
    Data  : logs_cologne8_direct    (5 seeds)

Usage:
    python plot_learning_curves.py
    python plot_learning_curves.py \
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


ALGO_COLORS  = {"dqn": "#1976D2", "ddqn": "#F57C00", "ppo": "#388E3C"}
ALGO_LABELS  = {"dqn": "DQN",    "ddqn": "DDQN",    "ppo": "PPO"}
REWARD_STYLES = {"queue": "-", "pressure": "--", "wait-clip": ":"}
REWARD_LABELS = {"queue": "Queue", "pressure": "Pressure", "wait-clip": "Wait-clip"}
REWARDS = ["queue", "pressure", "wait-clip"]

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      10,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "grid.linestyle": "--",
    "figure.dpi":     150,
})

LOG_RE = re.compile(r"step\s+(\d+)/\d+.*?mean_rew=([+-]?\d+\.?\d*(?:e[+-]?\d+)?)")


# ── Shared parse ──────────────────────────────────────────────────────────────

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


def load_logs(log_dir: Path):
    """Returns {algo: {reward: [(steps, rewards), ...]}}  — loads ALL seeds."""
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
        if algo not in ("dqn", "ddqn", "ppo"):
            continue
        steps, rewards = parse_log(f)
        if len(steps) >= 5:
            data[algo][reward].append((steps, rewards))
    total = sum(len(v) for d in data.values() for v in d.values())
    print(f"  Parsed {total} log files from {log_dir.name}/")
    return data


# ── C3 method: mean ± std with MAD outlier clipping  (plot_training_curves) ──

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


def smooth_mean(arr, w=5):
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def plot_c3_figure(data: dict, out: Path):
    """1x3 figure for C3 using mean ± std + MAD clipping (plot_training_curves method)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for col, reward in enumerate(REWARDS):
        ax = axes[col]
        ax.set_title(REWARD_LABELS[reward], fontsize=12, fontweight="bold")
        ax.set_xlabel("Timesteps", fontsize=9)
        if col == 0:
            ax.set_ylabel("Median Reward", fontsize=9)

        plotted = 0
        for algo in ("dqn", "ddqn", "ppo"):
            all_runs = data[algo].get(reward, [])
            if not all_runs:
                continue
            # Keep top-5 seeds by final-stage mean reward
            finals = np.array([_last_stage_mean_c8(np.array(s), np.array(r)) for s, r in all_runs])
            k = min(5, len(all_runs))
            idx = np.argsort(finals)[-k:]
            runs = [all_runs[i] for i in sorted(idx)]
            x, mean, std = interp_mean_std(runs)
            if len(mean) == 0:
                continue
            sm = smooth_mean(mean, w=5)
            ss = smooth_mean(std,  w=5)
            xs = x[:len(sm)]
            c  = ALGO_COLORS[algo]
            ax.plot(xs, sm, color=c, linewidth=2,
                    label=f"{ALGO_LABELS[algo]}  (n={len(runs)})")
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


# ── C8 method: top-5 seeds, median ± IQR, p5/p95 ylim  (commit 1b2961b) ──────

_RE_C8 = re.compile(r"step\s+(\d+)/\d+.*?mean_rew=([+-]?\d+(?:\.\d+)?)")

def _last_stage_mean_c8(s, r, cutoff=0.6):
    if len(s) == 0:
        return -np.inf
    mask = s >= s[-1] * cutoff
    return float(r[mask].mean()) if mask.any() else float(r[-5:].mean())


def collect_c8(log_dir: Path, reward: str, top_k: int = 5) -> dict:
    """Load logs, keep top-K seeds per algo (ranked by final-stage reward)."""
    data = {"dqn": [], "ddqn": [], "ppo": []}
    for f in sorted(log_dir.glob(f"*_{reward}_s*.log")):
        algo = f.stem.split("_")[0]
        if algo not in data:
            continue
        steps, rews = [], []
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = _RE_C8.search(line)
            if m:
                steps.append(int(m.group(1)))
                rews.append(float(m.group(2)))
        s, r = np.array(steps, dtype=float), np.array(rews, dtype=float)
        if len(s) >= 5:
            data[algo].append((s, r))
    for algo in list(data.keys()):
        runs = data[algo]
        if not runs:
            continue
        finals = np.array([_last_stage_mean_c8(s, r) for s, r in runs])
        k = min(top_k, len(runs))
        idx = np.argsort(finals)[-k:]
        data[algo] = [runs[i] for i in sorted(idx)]
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


def plot_c8_figure(log_dir: Path, out: Path):
    """1x3 figure for C8 — exact method from commit 1b2961b (top-5, p5/p95 ylim)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for col, reward in enumerate(REWARDS):
        ax = axes[col]
        ax.set_title(REWARD_LABELS[reward], fontsize=11, fontweight="bold")
        ax.set_xlabel("Timesteps", fontsize=9)
        if col == 0:
            ax.set_ylabel("Median Reward", fontsize=9)

        data    = collect_c8(log_dir, reward, top_k=5)
        any_run = False

        for algo in ("dqn", "ddqn", "ppo"):
            runs = data[algo]
            if not runs:
                continue
            x, med, q1, q3 = aggregate_c8(runs)
            if len(med) == 0:
                continue
            any_run = True
            ms  = smooth_c8(med, w=8)
            q1s = smooth_c8(q1,  w=8)
            q3s = smooth_c8(q3,  w=8)
            c = ALGO_COLORS[algo]
            ax.plot(x, ms, color=c, linewidth=1.8,
                    label=f"{ALGO_LABELS[algo]} (n={len(runs)})")
            ax.fill_between(x, q1s, q3s, color=c, alpha=0.18)

        ax.set_xlim(left=0)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", " "))
        )
        ax.tick_params(axis="x", labelsize=8)

        # ylim: p5–p95 of plotted lines, margin=25% of range, top capped at 0.02
        lines_data = [ln.get_ydata() for ln in ax.get_lines()
                      if len(ln.get_ydata()) > 1]
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
    p.add_argument("--c3_logs", default="./logs_cologne3_real_all",
                   help="C3: real route, no curriculum (10 seeds)")
    p.add_argument("--c8_logs", default="./logs_cologne8_direct",
                   help="C8: direct training, random route (5 seeds)")
    p.add_argument("--out",     default="./figures")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  plot_learning_curves.py")
    print(f"  C3 logs : {args.c3_logs}/  [mean+std, MAD clip]")
    print(f"  C8 logs : {args.c8_logs}/  [median+IQR]")
    print(f"  Output  : {args.out}/")
    print("=" * 60)

    print("\n--- Cologne3 (Real Route, no curriculum) ---")
    data_c3 = load_logs(Path(args.c3_logs))
    plot_c3_figure(data_c3, out_dir / "fig_training_c3.png")

    print("\n--- Cologne8 (Direct Training, Random Route) ---")
    plot_c8_figure(Path(args.c8_logs), out_dir / "fig_training_c8.png")

    print("\nHoan tat.")


if __name__ == "__main__":
    main()
