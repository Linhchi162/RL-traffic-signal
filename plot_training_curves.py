"""
plot_training_curves.py — Ve training curve tu log files cua DQN/DDQN/PPO.

Parse cac dong:
    DQN/DDQN: "  step  10000/500000 (  2.0%)  eps=...  mean_rew=+0.123"
    PPO:      "  step  10000/500000 (  2.0%)  mean_rew=+0.123"

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

ALGO_COLORS = {"dqn": "#1976D2", "ddqn": "#F57C00", "ppo": "#388E3C"}
ALGO_LABELS = {"dqn": "DQN",    "ddqn": "DDQN",    "ppo": "PPO"}
REWARD_LABELS = {"queue": "Queue", "pressure": "Pressure", "wait-clip": "Wait-clip"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})


def parse_log(log_path: Path):
    """Tra ve (steps, rewards) tu mot file log."""
    steps, rewards = [], []
    try:
        with open(log_path, errors="replace") as f:
            for line in f:
                m = LOG_RE.search(line)
                if m:
                    s = int(m.group(1))
                    r = float(m.group(2))
                    if not np.isnan(r):
                        steps.append(s)
                        rewards.append(r)
    except Exception:
        pass
    return steps, rewards


def smooth(arr, window=5):
    """Moving average smoothing."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def load_logs(log_dir: Path):
    """
    Returns: {algo: {reward: [(steps, rewards), ...]}}
    File naming: <algo>_<reward>_s<seed>.log
    """
    data = defaultdict(lambda: defaultdict(list))
    if not log_dir.exists():
        print(f"  [WARN] Khong tim thay log dir: {log_dir}")
        return data

    for f in sorted(log_dir.glob("*.log")):
        parts = f.stem.split("_")
        if len(parts) < 3:
            continue
        algo   = parts[0]
        seed   = parts[-1]
        reward = "_".join(parts[1:-1])
        if algo not in ("dqn", "ddqn", "ppo"):
            continue
        steps, rewards = parse_log(f)
        if len(steps) >= 2:
            data[algo][reward].append((steps, rewards))

    total = sum(len(runs) for rd in data.values() for runs in rd.values())
    print(f"  Parsed {total} log files from {log_dir.name}/")
    return data


def interp_to_common(runs, n_points=200):
    """Interpolate list of (steps, rewards) to common x-axis."""
    max_step = max(s[-1] for s, _ in runs if s)
    x = np.linspace(0, max_step, n_points)
    interped = []
    for steps, rewards in runs:
        if len(steps) < 2:
            continue
        interped.append(np.interp(x, steps, rewards))
    if not interped:
        return x, np.array([]), np.array([])
    arr  = np.array(interped)
    return x, arr.mean(axis=0), arr.std(axis=0)


def plot_scenario_curves(data: dict, scenario: str, out_dir: Path):
    """
    Mot figure per reward type.
    Trong moi figure: duong mean ± std cho DQN, DDQN, PPO.
    """
    all_rewards = sorted({rw for algo_d in data.values() for rw in algo_d})
    if not all_rewards:
        print(f"  [Skip] Khong co log nao cho {scenario}")
        return

    for rw in all_rewards:
        has_data = any(rw in data[a] and data[a][rw] for a in data)
        if not has_data:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_title(f"Training Curve — {scenario} | reward={REWARD_LABELS.get(rw, rw)}",
                     fontweight="bold")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Mean Step Reward")

        for algo in ("dqn", "ddqn", "ppo"):
            runs = data[algo].get(rw, [])
            if not runs:
                continue
            x, mean, std = interp_to_common(runs)
            if len(mean) == 0:
                continue
            mean_sm = smooth(mean, window=8)
            std_sm  = smooth(std,  window=8)
            xs      = x[:len(mean_sm)]
            c = ALGO_COLORS[algo]
            ax.plot(xs, mean_sm, color=c, label=f"{ALGO_LABELS[algo]} (n={len(runs)})",
                    linewidth=2)
            ax.fill_between(xs, mean_sm - std_sm, mean_sm + std_sm,
                            color=c, alpha=0.18)

        ax.legend()
        ax.set_xlim(left=0)
        fig.tight_layout()
        rw_safe = rw.replace("-", "_")
        fname   = f"fig_training_{scenario.lower()}_{rw_safe}.png"
        p = out_dir / fname
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p.name}")


def plot_all_rewards_combined(data: dict, scenario: str, out_dir: Path):
    """
    Mot figure: 3 subplots (queue / pressure / wait-clip), moi subplot co 3 algos.
    De de so sanh nhanh.
    """
    rewards_present = [rw for rw in ["queue", "pressure", "wait-clip"]
                       if any(rw in data[a] and data[a][rw] for a in data)]
    if not rewards_present:
        return

    fig, axes = plt.subplots(1, len(rewards_present),
                              figsize=(8 * len(rewards_present), 5),
                              sharey=False)
    if len(rewards_present) == 1:
        axes = [axes]
    fig.suptitle(f"Training Curves — {scenario}", fontsize=13, fontweight="bold")

    for ax, rw in zip(axes, rewards_present):
        ax.set_title(REWARD_LABELS.get(rw, rw))
        ax.set_xlabel("Timesteps")
        if ax is axes[0]:
            ax.set_ylabel("Mean Step Reward (smoothed)")

        for algo in ("dqn", "ddqn", "ppo"):
            runs = data[algo].get(rw, [])
            if not runs:
                continue
            x, mean, std = interp_to_common(runs)
            if len(mean) == 0:
                continue
            mean_sm = smooth(mean, window=8)
            std_sm  = smooth(std,  window=8)
            xs      = x[:len(mean_sm)]
            c = ALGO_COLORS[algo]
            ax.plot(xs, mean_sm, color=c, label=f"{ALGO_LABELS[algo]} (n={len(runs)})",
                    linewidth=2)
            ax.fill_between(xs, mean_sm - std_sm, mean_sm + std_sm,
                            color=c, alpha=0.15)
        ax.legend(fontsize=8)
        ax.set_xlim(left=0)

    fig.tight_layout()
    p = out_dir / f"fig_training_{scenario.lower()}_combined.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


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
    print(f"  C3 logs: {args.c3_logs}/")
    print(f"  C8 logs: {args.c8_logs}/")
    print(f"  Output : {args.out}/")
    print("=" * 55)

    for scenario, log_dir in [("Cologne3", Path(args.c3_logs)),
                               ("Cologne8", Path(args.c8_logs))]:
        print(f"\n--- {scenario} ---")
        data = load_logs(log_dir)
        if not any(data.values()):
            print(f"  Khong tim thay log files trong {log_dir}")
            continue
        plot_scenario_curves(data, scenario, out_dir)
        plot_all_rewards_combined(data, scenario, out_dir)

    print(f"\nHoan tat. Bieu do luu tai: {out_dir}/")


if __name__ == "__main__":
    main()
