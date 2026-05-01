"""
plot_learning_curves.py — Learning curves DQN / DDQN / PPO tren C3 va C8.

Doc training logs (StepLogger output), lay mean_rew theo training steps,
ve mean +- std qua seeds. Default dung reward=wait-clip.

Output: figures/fig_learning_curves.png  (2 panel: (a) C3, (b) C8)

Su dung:
    python plot_learning_curves.py
    python plot_learning_curves.py --reward queue --smooth 3
    python plot_learning_curves.py --reward all   # gop ca 3 reward type
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────────
ALGO_COLOR = {
    "dqn":  "#E91E63",   # pink
    "ddqn": "#1565C0",   # navy blue
    "ppo":  "#00BCD4",   # cyan
}
ALGO_LABEL = {"dqn": "DQN", "ddqn": "DDQN", "ppo": "PPO"}
REWARDS    = ["queue", "pressure", "wait-clip"]

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      10,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "grid.linestyle": "--",
    "figure.dpi":     150,
})

# ── Parsing ───────────────────────────────────────────────────────────────────
_RE = re.compile(r"step\s+(\d+)/\d+.*?mean_rew=([+-]?\d+\.\d+)")

def parse_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps, rews = [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _RE.search(line)
        if m:
            steps.append(int(m.group(1)))
            rews.append(float(m.group(2)))
    return np.array(steps, dtype=float), np.array(rews, dtype=float)


def collect_algo(log_dir: Path, reward: str) -> dict[str, list[tuple]]:
    """
    Returns {algo: [(steps, rews), ...]} for all seeds.
    reward='all' merges all 3 reward types per algo.
    """
    data: dict[str, list] = {"dqn": [], "ddqn": [], "ppo": []}
    rewlist = REWARDS if reward == "all" else [reward]
    for rw in rewlist:
        for f in sorted(log_dir.glob(f"*_{rw}_s*.log")):
            algo = f.stem.split("_")[0]
            if algo not in data:
                continue
            s, r = parse_log(f)
            if len(s) >= 5:
                data[algo].append((s, r))
    return data


# ── Aggregate: align seeds at shared step grid ────────────────────────────────

def aggregate(runs: list[tuple]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align all runs to a shared step grid (union of all step positions),
    interpolating missing values. Returns (steps_k, mean, std).
    Values are negated so higher = better (learning progress going up).
    """
    if not runs:
        return np.array([]), np.array([]), np.array([])

    # Build common step grid from first run (all seeds should match)
    all_steps = sorted({int(v) for s, _ in runs for v in s})
    grid = np.array(all_steps, dtype=float)

    mat = np.array([
        np.interp(grid, s, -r)      # negate: high start, drop = learning
        for s, r in runs
    ])

    mean = mat.mean(axis=0)
    std  = mat.std(axis=0, ddof=1) if len(mat) > 1 else np.zeros_like(mean)
    return grid / 1_000, mean, std   # x in units of 1000 steps


def smooth(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


# ── Panel ─────────────────────────────────────────────────────────────────────

def plot_panel(ax, log_dir: Path, reward: str, smooth_w: int,
               title: str, ylabel: bool):
    data = collect_algo(log_dir, reward)

    for algo in ("dqn", "ddqn", "ppo"):
        runs = data[algo]
        if not runs:
            continue

        color = ALGO_COLOR[algo]

        # Individual seed lines (thin, transparent)
        all_x = None
        for s, r in runs:
            x_k = s / 1_000
            y   = smooth(-r, smooth_w)
            ax.plot(x_k, y, color=color, linewidth=0.4, alpha=0.25)
            all_x = x_k   # same for all seeds

        # Thick mean line
        x_k, mean, _ = aggregate(runs)
        if len(mean) == 0:
            continue
        ms = smooth(mean, smooth_w)
        ax.plot(x_k, ms, color=color, linewidth=2.2,
                label=ALGO_LABEL[algo])

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Training steps (×10³)", fontsize=10)
    if ylabel:
        ax.set_ylabel("Negated reward (high→bad, low→good)", fontsize=9)
    ax.set_xlim(0, None)
    ax.set_ylim(bottom=0)

    n = max((len(v) for v in data.values()), default=0)
    rw_label = reward if reward != "all" else "all rewards"
    ax.text(0.98, 0.96, f"reward: {rw_label}   n={n} runs/algo",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color="#666666")
    ax.legend(fontsize=10, loc="best", framealpha=0.85)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--c3_logs", default="./logs_cologne3_real_all")
    p.add_argument("--c8_logs", default="./logs_cologne8_real_all")
    p.add_argument("--reward",  default="wait-clip",
                   choices=["queue", "pressure", "wait-clip", "all"])
    p.add_argument("--smooth",  type=int, default=8,
                   help="Smoothing window over the aligned step grid.")
    p.add_argument("--out",     default="./figures")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    plot_panel(axes[0], Path(args.c3_logs), args.reward, args.smooth,
               title="(a) Cologne3",  ylabel=True)
    plot_panel(axes[1], Path(args.c8_logs), args.reward, args.smooth,
               title="(b) Cologne8",  ylabel=False)

    fig.suptitle("Learning Curves: DQN vs DDQN vs PPO\n"
                 "(thin lines = individual seeds, thick = mean)",
                 fontsize=11, y=1.01)
    fig.tight_layout()

    out = out_dir / "fig_learning_curves.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
