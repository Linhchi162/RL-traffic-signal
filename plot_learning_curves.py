"""
plot_learning_curves.py — Training reward curves, layout 2x3.

Row 0 : Cologne3   |  Row 1 : Cologne8
Col 0 : Queue      |  Col 1 : Pressure  |  Col 2 : Wait-clip

Moi panel ve mean +- std (shaded band) qua seeds.

Su dung:
    python plot_learning_curves.py
    python plot_learning_curves.py --c3_logs logs_cologne3_rand_curr \
        --c8_logs logs_cologne8_rand_curr --smooth 8 --out figures
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────────
ALGO_COLOR = {"dqn": "#1f77b4", "ddqn": "#ff7f0e", "ppo": "#2ca02c"}
ALGO_LABEL = {"dqn": "DQN", "ddqn": "DDQN", "ppo": "PPO"}
REWARDS       = ["queue", "pressure", "wait-clip"]
REWARD_TITLES = {"queue": "Queue", "pressure": "Pressure", "wait-clip": "Wait-clip"}

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      10,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "grid.linestyle": "--",
    "figure.dpi":     150,
})

# ── Parsing ───────────────────────────────────────────────────────────────────
_RE = re.compile(r"step\s+(\d+)/\d+.*?mean_rew=([+-]?\d+(?:\.\d+)?)")

def parse_log(path: Path):
    steps, rews = [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _RE.search(line)
        if m:
            steps.append(int(m.group(1)))
            rews.append(float(m.group(2)))
    return np.array(steps, dtype=float), np.array(rews, dtype=float)


def collect_algo(log_dir: Path, reward: str) -> dict:
    """Returns {algo: [(steps, rews), ...]} for the given reward type."""
    data: dict = {"dqn": [], "ddqn": [], "ppo": []}
    for f in sorted(log_dir.glob(f"*_{reward}_s*.log")):
        algo = f.stem.split("_")[0]
        if algo not in data:
            continue
        s, r = parse_log(f)
        if len(s) >= 5:
            data[algo].append((s, r))
    return data


# ── Aggregate ─────────────────────────────────────────────────────────────────

def aggregate(runs):
    if not runs:
        return np.array([]), np.array([]), np.array([])
    all_steps = sorted({int(v) for s, _ in runs for v in s})
    grid = np.array(all_steps, dtype=float)
    mat  = np.array([np.interp(grid, s, r) for s, r in runs])
    mean = mat.mean(axis=0)
    std  = mat.std(axis=0, ddof=1) if len(mat) > 1 else np.zeros_like(mean)
    return grid, mean, std


def smooth(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="same")


# ── Panel ─────────────────────────────────────────────────────────────────────

def plot_panel(ax, log_dir: Path, reward: str, smooth_w: int,
               col_title: str, row_label: str):
    data = collect_algo(log_dir, reward)

    for algo in ("dqn", "ddqn", "ppo"):
        runs = data[algo]
        if not runs:
            continue
        color = ALGO_COLOR[algo]
        x, mean, std = aggregate(runs)
        if len(mean) == 0:
            continue

        ms = smooth(mean, smooth_w)
        ss = smooth(std,  smooth_w)

        label = f"{ALGO_LABEL[algo]} (n={len(runs)})"
        ax.plot(x, ms, color=color, linewidth=1.8, label=label)
        ax.fill_between(x, ms - ss, ms + ss, color=color, alpha=0.2)

    if col_title:
        ax.set_title(col_title, fontsize=11, fontweight="bold")

    ax.set_xlabel("Timesteps", fontsize=9)

    if row_label:
        ax.set_ylabel(f"{row_label}\nMean Reward (smoothed)", fontsize=9)

    # Format x-axis: show full numbers (100000, 200000 ...)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", " "))
    )
    ax.tick_params(axis="x", labelsize=8)

    ax.legend(fontsize=8, loc="best", framealpha=0.85)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--c3_logs", default="./logs_cologne3_rand_curr")
    p.add_argument("--c8_logs", default="./logs_cologne8_rand_curr")
    p.add_argument("--smooth",  type=int, default=8)
    p.add_argument("--out",     default="./figures")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        (Path(args.c3_logs), "Cologne3"),
        (Path(args.c8_logs), "Cologne8"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Training Reward Curves", fontsize=13, fontweight="bold", y=1.01)

    for row, (log_dir, scenario_name) in enumerate(scenarios):
        for col, reward in enumerate(REWARDS):
            ax        = axes[row][col]
            col_title = REWARD_TITLES[reward] if row == 0 else ""
            row_label = scenario_name         if col == 0 else ""
            plot_panel(ax, log_dir, reward, args.smooth, col_title, row_label)

    fig.tight_layout()

    out = out_dir / "fig_training_curves.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
