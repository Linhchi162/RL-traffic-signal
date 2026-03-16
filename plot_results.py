"""
plot_results.py — Ve bieu do so sanh ket qua (Tai hien Fig 5 trong bai bao RLTSCQ)

Output:
    results/plots/fig5a_queue_vs_cycle.png
    results/plots/fig5b_high_flow.png
    results/plots/fig5c_medium_flow.png
    results/plots/fig5d_low_flow.png
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Fix encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

def set_style():
    """Cai dat style bieu do cho giong bai bao."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'savefig.bbox': 'tight'
    })

# Mapping ten cac model de hien thi dep hon
NAME_MAP = {
    "ppo_queue": "PPO (Queue)",
    "ppo_diff-waiting-time": "PPO (Delay)",
    "ppo_average-speed": "PPO (Speed)",
    "ppo_pressure": "PPO (Pressure)",
    "ppo_queue-AE-4d": "PPO (Queue AE-4D)",
    "ppo_queue-AE-8d": "PPO (Queue AE-8D)",
    "ppo_queue-AE-16d": "PPO (Queue AE-16D)",
    "ppo_queue-AE-19d": "PPO (Queue AE-19D)",
    "ppo_queue-AE-32d": "PPO (Queue AE-32D)",
    "dqn_wait-clip": "RESCO-DQN",
    "webster_-": "Dynamic Webster",
    "fixed_-": "Fixed-time Baseline"
}

COLOR_MAP = {
    "PPO (Queue)": "#e41a1c",
    "PPO (Delay)": "#377eb8",
    "PPO (Speed)": "#4daf4a",
    "PPO (Pressure)": "#984ea3",
    "PPO (Queue AE-4D)": "#a65628",
    "PPO (Queue AE-8D)": "#f781bf",
    "PPO (Queue AE-16D)": "#e41a1c",  # Main
    "PPO (Queue AE-19D)": "#999999",
    "PPO (Queue AE-32D)": "#a6d854",
    "RESCO-DQN": "#ff7f00",
    "Dynamic Webster": "#808080",
    "Fixed-time Baseline": "#000000"
}


def plot_bar_chart(summary_df, flow_condition, title, out_path):
    """
    Ve bieu do cot the hien hang cho trung binh + error bars (tuong ung Fig 5b, 5c, 5d).
    """
    df = summary_df[summary_df["flow"] == flow_condition].copy()
    if df.empty:
        print(f"  [WARN] Khong co du lieu cho flow={flow_condition}")
        return

    # Tao ten model hien thi
    df["model_name"] = df.apply(lambda r: f"{r['algo']}_{r['reward']}", axis=1)
    df["display_name"] = df["model_name"].map(lambda x: NAME_MAP.get(x, x))

    # Sap xep theo ten
    df = df.sort_values("display_name")

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(df))
    means = df["mean_queue"].values
    stds = df["std_queue"].values
    labels = df["display_name"].values

    colors = [COLOR_MAP.get(l, "#cccccc") for l in labels]

    ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.9, ecolor='black', capsize=8, color=colors)

    ax.set_ylabel('Mean Queue Length (Vehicles)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(title)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results", help="Thu muc chua ket qua tu evaluate_all.py")
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    plots_dir = res_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    summary_file = res_dir / "summary.csv"
    if not summary_file.exists():
        sys.exit(f"Khong tim thay {summary_file}. Hay chay evaluate_all.py truoc.")

    df_sum = pd.read_csv(summary_file)
    set_style()

    print("=" * 55)
    print("  VE BIEU DO KET QUA (Fig 5)")
    print("=" * 55)

    # Fig 5b: High Flow
    out_b = plots_dir / "fig5b_high_flow.png"
    plot_bar_chart(df_sum, "high", "Total Queue Length - High Flow Conditions", out_b)
    print(f"  [OK] Fig 5b : {out_b.name}")

    # Fig 5c: Medium Flow
    out_c = plots_dir / "fig5c_medium_flow.png"
    plot_bar_chart(df_sum, "medium", "Total Queue Length - Medium Flow Conditions", out_c)
    print(f"  [OK] Fig 5c : {out_c.name}")

    # Fig 5d: Low Flow
    out_d = plots_dir / "fig5d_low_flow.png"
    plot_bar_chart(df_sum, "low", "Total Queue Length - Low Flow Conditions", out_d)
    print(f"  [OK] Fig 5d : {out_d.name}")

    # Ghi chu ve Fig 5a
    print("\n  [INFO] Fig 5a (Line chart theo tung chu ky) de ve duoc can thu thap ")
    print("         du lieu timestep tu `evaluate_ppo.py` ban goc (da log ra `cycle_log.csv`).")
    print("         De giu cho pipeline nhanh gon, Script hien tai tap trung ve cac Bar Chart.")

    print("=" * 55)
    print(f"Hoan tat: Thu muc {plots_dir}/")


if __name__ == "__main__":
    main()
