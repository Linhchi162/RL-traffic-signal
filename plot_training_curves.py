"""
plot_training_curves.py — Ve training curve tu log files.

1 figure per scenario (Cologne3, Cologne8).
Moi figure: tat ca 9 duong (3 algo x 3 reward) tren cung 1 truc.
  - Mau = algo:   DQN=blue  DDQN=orange  PPO=green
  - Net  = reward: queue=solid  pressure=dashed  wait-clip=dotted

Su dung:
    python plot_training_curves.py
    python plot_training_curves.py --c3_logs logs_cologne3_real_all \
                                    --c8_logs logs_cologne8_real_all \
                                    --out figures
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG_RE = re.compile(r"step\s+(\d+)/\d+.*?mean_rew=([+-]?\d+\.?\d*(?:e[+-]?\d+)?)")

ALGO_COLORS  = {"dqn": "#1976D2", "ddqn": "#F57C00", "ppo": "#388E3C"}
ALGO_LABELS  = {"dqn": "DQN",    "ddqn": "DDQN",    "ppo": "PPO"}
REWARD_STYLES = {"queue": "-", "pressure": "--", "wait-clip": ":"}
REWARD_LABELS = {"queue": "Queue", "pressure": "Pressure", "wait-clip": "Wait-clip"}

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      10,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "grid.linestyle": "--",
    "figure.dpi":     150,
})


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


def smooth(arr, w=10):
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def load_logs(log_dir: Path):
    """Returns {algo: {reward: [(steps, rewards), ...]}}"""
    data = defaultdict(lambda: defaultdict(list))
    if not log_dir.exists():
        print(f"  [WARN] Khong tim thay: {log_dir}")
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


def interp_mean_std(runs, n_pts=200):
    max_step = max(s[-1] for s, _ in runs if s)
    x = np.linspace(0, max_step, n_pts)
    arr = np.array([np.interp(x, s, r) for s, r in runs if len(s) >= 2])
    if arr.ndim == 1 or len(arr) == 0:
        return x, np.array([]), np.array([])
    return x, arr.mean(axis=0), arr.std(axis=0)


def plot_grid(data_c3: dict, data_c8: dict, out: Path):
    """
    Luoi 2×3:
      Hang: Cologne3 (tren) | Cologne8 (duoi)
      Cot : Queue | Pressure | Wait-clip
    Moi o: 3 lines DQN / DDQN / PPO, mean ± std shading.
    """
    SCENARIOS = [("Cologne3", data_c3), ("Cologne8", data_c8)]
    REWARD_ORDER = ["queue", "pressure", "wait-clip"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=False)
    fig.suptitle("Training Reward Curves", fontsize=14, fontweight="bold", y=1.01)

    for row_i, (scenario, data) in enumerate(SCENARIOS):
        for col_i, reward in enumerate(REWARD_ORDER):
            ax = axes[row_i][col_i]

            # Column header (chi hang dau)
            if row_i == 0:
                ax.set_title(REWARD_LABELS[reward], fontsize=12, fontweight="bold")

            # Row label (chi cot dau)
            if col_i == 0:
                ax.set_ylabel(f"{scenario}\nMean Reward (smoothed)", fontsize=9)

            if col_i == 1:
                ax.set_xlabel("Timesteps", fontsize=9)

            plotted = 0
            for algo in ("dqn", "ddqn", "ppo"):
                runs = data[algo].get(reward, [])
                if not runs:
                    continue
                x, mean, std = interp_mean_std(runs)
                if len(mean) == 0:
                    continue
                sm = smooth(mean, w=5)
                ss = smooth(std,  w=5)
                xs = x[:len(sm)]
                c  = ALGO_COLORS[algo]
                ax.plot(xs, sm, color=c, linewidth=2,
                        label=f"{ALGO_LABELS[algo]}  (n={len(runs)})")
                ax.fill_between(xs, sm - ss, sm + ss, color=c, alpha=0.15)
                plotted += 1

            ax.set_xlim(left=0)
            ax.grid(True, alpha=0.3, linestyle="--")
            if plotted > 0:
                ax.legend(fontsize=7.5, loc="lower right", framealpha=0.8)
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="gray", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--c3_logs", default="./logs_cologne3_real_all")
    p.add_argument("--c8_logs", default="./logs_cologne8_real_all")
    p.add_argument("--out",     default="./figures")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  plot_training_curves.py")
    print(f"  C3 logs : {args.c3_logs}/")
    print(f"  C8 logs : {args.c8_logs}/")
    print(f"  Output  : {args.out}/")
    print("=" * 55)

    print("\n--- Cologne3 ---")
    data_c3 = load_logs(Path(args.c3_logs))
    print("\n--- Cologne8 ---")
    data_c8 = load_logs(Path(args.c8_logs))

    out = out_dir / "fig_training_grid.png"
    plot_grid(data_c3, data_c8, out)

    print(f"\nHoan tat.")


if __name__ == "__main__":
    main()
