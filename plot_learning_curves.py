"""
plot_learning_curves.py — Training reward curves, layout 2x3.

Row 0 : Cologne3   |  Row 1 : Cologne8
Col 0 : Queue      |  Col 1 : Pressure  |  Col 2 : Wait-clip

Moi panel ve median + Q1/Q3 band qua seeds.
Duong doc: cac buoc chuyen curriculum stage.

Su dung:
    python plot_learning_curves.py
    python plot_learning_curves.py \
        --c3_logs logs_cologne3_rand_curr \
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
TOP_K_SEEDS   = 5   # chi giu top-K seed tot nhat moi (algo, reward)

# Curriculum stage boundaries (fraction of total_steps=500k)
CURRICULUM     = [0.25, 0.50, 0.75, 1.00]
STAGE_PORTIONS = [0.15, 0.20, 0.25, 0.40]
TOTAL_STEPS    = 500_000
STAGE_STARTS   = []
_cum = 0
for portion in STAGE_PORTIONS[:-1]:
    _cum += round(TOTAL_STEPS * portion)
    STAGE_STARTS.append(_cum)   # [75000, 175000, 300000]

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


def _last_stage_mean(s: np.ndarray, r: np.ndarray,
                     cutoff: float = 0.6) -> float:
    """Mean reward in the last (1-cutoff) fraction of training."""
    if len(s) == 0:
        return -np.inf
    threshold = s[-1] * cutoff
    mask = s >= threshold
    return float(r[mask].mean()) if mask.any() else float(r[-5:].mean())


def collect_algo(log_dir: Path, reward: str) -> dict:
    data: dict = {"dqn": [], "ddqn": [], "ppo": []}
    for f in sorted(log_dir.glob(f"*_{reward}_s*.log")):
        algo = f.stem.split("_")[0]
        if algo not in data:
            continue
        s, r = parse_log(f)
        if len(s) >= 5:
            data[algo].append((s, r))

    # Keep top-K seeds by last-stage mean reward (higher = better)
    for algo in list(data.keys()):
        runs = data[algo]
        if not runs:
            continue
        finals = np.array([_last_stage_mean(s, r) for s, r in runs])
        top_k  = min(TOP_K_SEEDS, len(runs))
        idx    = np.argsort(finals)[-top_k:]          # highest reward = best
        data[algo] = [runs[i] for i in sorted(idx)]

    return data


# ── Aggregate: median + Q1/Q3 ─────────────────────────────────────────────────

def aggregate(runs):
    """Returns (grid, median, q1, q3) — robust to outlier seeds."""
    if not runs:
        return np.array([]), np.array([]), np.array([]), np.array([])
    all_steps = sorted({int(v) for s, _ in runs for v in s})
    grid = np.array(all_steps, dtype=float)
    mat  = np.array([np.interp(grid, s, r) for s, r in runs])
    med  = np.median(mat, axis=0)
    q1   = np.percentile(mat, 25, axis=0)
    q3   = np.percentile(mat, 75, axis=0)
    return grid, med, q1, q3


def smooth(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="same")


# ── Panel ─────────────────────────────────────────────────────────────────────

def plot_panel(ax, log_dir: Path, reward: str, smooth_w: int,
               col_title: str, row_label: str):
    data    = collect_algo(log_dir, reward)
    any_run = False

    for algo in ("dqn", "ddqn", "ppo"):
        runs = data[algo]
        if not runs:
            continue
        color = ALGO_COLOR[algo]
        x, med, q1, q3 = aggregate(runs)
        if len(med) == 0:
            continue
        any_run = True

        ms  = smooth(med, smooth_w)
        q1s = smooth(q1,  smooth_w)
        q3s = smooth(q3,  smooth_w)

        label = f"{ALGO_LABEL[algo]} (n={len(runs)})"
        ax.plot(x, ms, color=color, linewidth=1.8, label=label)
        ax.fill_between(x, q1s, q3s, color=color, alpha=0.18)

    # Curriculum stage transition lines
    for stage_x in STAGE_STARTS:
        ax.axvline(stage_x, color="gray", linestyle=":", linewidth=0.9, alpha=0.7)

    if col_title:
        ax.set_title(col_title, fontsize=11, fontweight="bold")

    ax.set_xlabel("Timesteps", fontsize=9)

    if row_label:
        ax.set_ylabel(f"{row_label}\nMedian Reward (smoothed)", fontsize=9)

    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", " "))
    )
    ax.tick_params(axis="x", labelsize=8)

    if any_run:
        ax.legend(fontsize=8, loc="lower right", framealpha=0.85)


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
    fig.suptitle(
        "Training Reward Curves  (median ± IQR, dotted lines = curriculum stage)",
        fontsize=12, fontweight="bold", y=1.01
    )

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
