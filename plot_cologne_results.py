"""
plot_cologne_results.py — Ve bieu do thesis cho Cologne3 & Cologne8.

Output:
    fig1a_queue.png        — Queue bar chart, C3 | C8 subplots
    fig1b_wait.png         — Wait/veh bar chart, C3 | C8 subplots
    fig2_scalability.png   — Scalability grouped bar (best-per-algo, queue)
    fig3_ablation.png      — Reward ablation: C3 | C8 subplots
    fig4_boxplot.png       — Seed stability boxplot: C3 | C8 subplots
    improvement_vs_baselines.csv  (luu cho reproducibility)

Su dung:
    python plot_cologne_results.py
    python plot_cologne_results.py --c3 results_cologne3_real \
                                    --c8 results_cologne8_real --out figures
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Style ────────────────────────────────────────────────────────────────────
ALGO_COLORS = {
    "dqn":     "#1976D2",
    "ddqn":    "#F57C00",
    "ppo":     "#388E3C",
    "fixed":   "#757575",
    "webster": "#455A64",
}
ALGO_LABELS = {
    "dqn": "DQN", "ddqn": "DDQN", "ppo": "PPO",
    "fixed": "Fixed-time", "webster": "Webster",
}
REWARD_LABELS = {"queue": "Queue", "pressure": "Pressure", "wait-clip": "Wait-clip"}
REWARD_HATCH  = {"queue": "", "pressure": "//", "wait-clip": "xx"}

BASELINES  = ["fixed", "webster"]          # bỏ random
RL_ALGOS   = ["dqn", "ddqn", "ppo"]
REWARDS    = ["queue", "pressure", "wait-clip"]

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi":      150,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "grid.linestyle":  "--",
})


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"  [WARN] Khong tim thay: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str)
    for col in ["mean_queue", "mean_wait", "mean_wait_per_veh", "mean_speed",
                "throughput", "mean_travel_time", "total_reward"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def rdf(df): return df[~df["algo"].isin(["random", "fixed", "webster"])].copy()
def bdf(df): return df[ df["algo"].isin(BASELINES)].copy()


# ── Helper: bar groups cho 1 scenario ────────────────────────────────────────

def build_bar_groups(df: pd.DataFrame, col: str):
    """
    Returns list of (label, mean, std, color, hatch).
    Bao gom 9 RL combos (algo x reward) + Fixed + Webster (khong co Random).
    """
    r_ = rdf(df); b_ = bdf(df)
    groups = []
    for algo in RL_ALGOS:
        for rw in REWARDS:
            sub = r_[(r_["algo"] == algo) & (r_["reward"] == rw)][col].dropna()
            if sub.empty:
                continue
            lbl = f"{ALGO_LABELS[algo]}\n{REWARD_LABELS[rw]}"
            groups.append((lbl, sub.mean(), sub.std(),
                            ALGO_COLORS[algo], REWARD_HATCH[rw]))
    for bl in BASELINES:
        sub = b_[b_["algo"] == bl][col].dropna()
        if sub.empty:
            continue
        groups.append((ALGO_LABELS[bl], sub.mean(), 0, ALGO_COLORS[bl], ""))
    return groups


def draw_bars(ax, groups, ylabel, show_separator=True):
    x = np.arange(len(groups))
    for i, (_, m, s, c, h) in enumerate(groups):
        ax.bar(x[i], m, yerr=s if s > 0 else None,
               capsize=3, color=c, hatch=h, alpha=0.85,
               edgecolor="gray" if h else "white", linewidth=0.5)
    n_rl = len(RL_ALGOS) * len(REWARDS)
    if show_separator and n_rl < len(groups):
        ax.axvline(n_rl - 0.5, color="black", lw=1.1, ls="--", alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)


# ── Figure 1a: Queue bar chart ────────────────────────────────────────────────

def fig1a_queue(df3: pd.DataFrame, df8: pd.DataFrame, out: Path):
    col = "mean_queue"
    scenarios = [(n, d) for n, d in [("Cologne3", df3), ("Cologne8", df8)] if not d.empty]
    if not scenarios:
        return

    fig, axes = plt.subplots(1, len(scenarios), figsize=(9 * len(scenarios), 5.5),
                              sharey=False)
    if len(scenarios) == 1:
        axes = [axes]
    fig.suptitle("Mean Queue Length Comparison", fontsize=13, fontweight="bold", y=1.02)

    for ax, (name, df) in zip(axes, scenarios):
        if col not in df.columns:
            continue
        groups = build_bar_groups(df, col)
        draw_bars(ax, groups, "Mean Queue (veh)")
        ax.set_title(name, fontsize=11)

    _add_legend(fig)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── Figure 1b: Wait/veh bar chart ────────────────────────────────────────────

def fig1b_wait(df3: pd.DataFrame, df8: pd.DataFrame, out: Path):
    col = "mean_wait_per_veh"
    scenarios = [(n, d) for n, d in [("Cologne3", df3), ("Cologne8", df8)] if not d.empty]
    if not scenarios:
        return

    fig, axes = plt.subplots(1, len(scenarios), figsize=(9 * len(scenarios), 5.5),
                              sharey=False)
    if len(scenarios) == 1:
        axes = [axes]
    fig.suptitle("Mean Waiting Time per Vehicle", fontsize=13, fontweight="bold", y=1.02)

    for ax, (name, df) in zip(axes, scenarios):
        if col not in df.columns:
            continue
        groups = build_bar_groups(df, col)
        draw_bars(ax, groups, "Mean Wait/veh (s)")
        ax.set_title(name, fontsize=11)

    _add_legend(fig)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def _add_legend(fig):
    """Legend chung: mau = algo, hatch = reward type."""
    handles = (
        [plt.Rectangle((0,0),1,1, fc=ALGO_COLORS[a], label=ALGO_LABELS[a])
         for a in RL_ALGOS + BASELINES]
        + [plt.Rectangle((0,0),1,1, fc="#BBBBBB", hatch=REWARD_HATCH[rw], ec="gray",
                         label=f"reward={REWARD_LABELS[rw]}") for rw in REWARDS]
    )
    fig.legend(handles=handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.9, fontsize=7.5)


# ── Figure 2: Scalability C3 → C8 ────────────────────────────────────────────

def fig2_scalability(df3: pd.DataFrame, df8: pd.DataFrame, out: Path):
    """
    Grouped bar chart: x = algo (best reward per algo) + Fixed + Webster.
    2 bars per group: C3 (solid) | C8 (hatched).
    Metric: queue only.
    """
    if df3.empty and df8.empty:
        return
    col = "mean_queue"

    items = []   # (label, c3_mean, c3_std, c8_mean, c8_std, color)

    for algo in RL_ALGOS:
        r3 = rdf(df3)[rdf(df3)["algo"] == algo] if not df3.empty else pd.DataFrame()
        r8 = rdf(df8)[rdf(df8)["algo"] == algo] if not df8.empty else pd.DataFrame()

        # Best reward theo C3 (hoac C8 neu C3 khong co)
        src = r3 if not r3.empty else r8
        if src.empty:
            continue
        grp = src.groupby("reward")[col].mean()
        best_r = grp.idxmin()

        v3 = r3[r3["reward"] == best_r][col].dropna() if not r3.empty else pd.Series(dtype=float)
        v8 = r8[r8["reward"] == best_r][col].dropna() if not r8.empty else pd.Series(dtype=float)

        lbl = f"{ALGO_LABELS[algo]}\n({REWARD_LABELS.get(best_r, best_r)})"
        items.append((lbl,
                      v3.mean() if not v3.empty else np.nan, v3.std() if not v3.empty else 0,
                      v8.mean() if not v8.empty else np.nan, v8.std() if not v8.empty else 0,
                      ALGO_COLORS[algo]))

    for bl in BASELINES:
        b3 = bdf(df3)[bdf(df3)["algo"] == bl][col].dropna() if not df3.empty else pd.Series(dtype=float)
        b8 = bdf(df8)[bdf(df8)["algo"] == bl][col].dropna() if not df8.empty else pd.Series(dtype=float)
        if b3.empty and b8.empty:
            continue
        items.append((ALGO_LABELS[bl],
                      b3.mean() if not b3.empty else np.nan, 0,
                      b8.mean() if not b8.empty else np.nan, 0,
                      ALGO_COLORS[bl]))

    if not items:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(items) * 1.6), 5))
    ax.set_title("Scalability: Cologne3 → Cologne8  (queue length)",
                 fontsize=12, fontweight="bold")

    n = len(items); x = np.arange(n); w = 0.38
    for i, (lbl, m3, s3, m8, s8, c) in enumerate(items):
        if not np.isnan(m3):
            ax.bar(x[i] - w/2, m3, w, yerr=s3 if s3 > 0 else None,
                   capsize=3, color=c, alpha=0.92,
                   label="Cologne3" if i == 0 else "_")
        if not np.isnan(m8):
            ax.bar(x[i] + w/2, m8, w, yerr=s8 if s8 > 0 else None,
                   capsize=3, color=c, alpha=0.42, hatch="//",
                   label="Cologne8" if i == 0 else "_")

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in items], rotation=20, ha="right")
    ax.set_ylabel("Mean Queue (veh)")
    ax.legend(["Cologne3", "Cologne8 (hatched)"], fontsize=9)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── Figure 3: Reward ablation (C3 | C8 subplots) ─────────────────────────────

def fig3_ablation(df3: pd.DataFrame, df8: pd.DataFrame, out: Path):
    """
    2 rows (C3, C8) × 3 cols (DQN, DDQN, PPO).
    Y-axis: queue length.
    3 bars per subplot: queue / pressure / wait-clip.
    """
    REWARD_COLORS = {"queue": "#1565C0", "pressure": "#6A1B9A", "wait-clip": "#BF360C"}
    col = "mean_queue"

    scenarios = [(n, d) for n, d in [("Cologne3", df3), ("Cologne8", df8)]
                 if not d.empty and col in d.columns]
    if not scenarios:
        return

    n_rows = len(scenarios)
    fig, axes = plt.subplots(n_rows, len(RL_ALGOS),
                              figsize=(4.5 * len(RL_ALGOS), 4 * n_rows),
                              squeeze=False)
    fig.suptitle("Reward Function Ablation (Queue Length)",
                 fontsize=12, fontweight="bold", y=1.02)

    for row_i, (scenario, df) in enumerate(scenarios):
        r_ = rdf(df)
        for col_i, algo in enumerate(RL_ALGOS):
            ax = axes[row_i][col_i]
            sub = r_[r_["algo"] == algo]
            means, stds, lbls, clrs = [], [], [], []
            for rw in REWARDS:
                vals = sub[sub["reward"] == rw][col].dropna()
                if vals.empty:
                    continue
                means.append(vals.mean())
                stds.append(vals.std())
                lbls.append(REWARD_LABELS[rw])
                clrs.append(REWARD_COLORS[rw])
            if not means:
                continue
            x = np.arange(len(means))
            bars = ax.bar(x, means, yerr=stds, capsize=4, color=clrs, alpha=0.85)
            # Highlight best
            best_i = int(np.argmin(means))
            bars[best_i].set_edgecolor("black"); bars[best_i].set_linewidth(2)
            ax.set_xticks(x); ax.set_xticklabels(lbls)
            if col_i == 0:
                ax.set_ylabel(f"{scenario}\nMean Queue (veh)")
            if row_i == 0:
                ax.set_title(ALGO_LABELS[algo], fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── Figure 4: Boxplot seed stability (C3 | C8) ───────────────────────────────

def fig4_boxplot(df3: pd.DataFrame, df8: pd.DataFrame, out: Path):
    """
    2 subplots: C3 | C8.
    X-axis: algo × reward combos.
    Baseline horizontal lines: Fixed-time, Webster.
    """
    col = "mean_queue"
    line_styles = {"fixed": "--", "webster": "-."}

    scenarios = [(n, d) for n, d in [("Cologne3", df3), ("Cologne8", df8)]
                 if not d.empty and col in d.columns]
    if not scenarios:
        return

    fig, axes = plt.subplots(1, len(scenarios),
                              figsize=(8 * len(scenarios), 5.5), sharey=False)
    if len(scenarios) == 1:
        axes = [axes]
    fig.suptitle("Seed Stability — Mean Queue (10 seeds)", fontsize=12, fontweight="bold")

    for ax, (scenario, df) in zip(axes, scenarios):
        r_ = rdf(df); b_ = bdf(df)
        data_list, tick_labels, tick_colors = [], [], []

        for algo in RL_ALGOS:
            for rw in REWARDS:
                vals = r_[(r_["algo"] == algo) & (r_["reward"] == rw)][col].dropna()
                if vals.empty:
                    continue
                data_list.append(vals.values)
                tick_labels.append(f"{ALGO_LABELS[algo]}\n{REWARD_LABELS[rw]}")
                tick_colors.append(ALGO_COLORS[algo])

        if not data_list:
            continue

        bp = ax.boxplot(data_list, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2),
                        flierprops=dict(marker="o", markersize=3, alpha=0.5))
        for patch, c in zip(bp["boxes"], tick_colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)

        for bl in BASELINES:
            bval = b_[b_["algo"] == bl][col].mean()
            if not np.isnan(bval):
                ax.axhline(bval, color=ALGO_COLORS[bl],
                           linestyle=line_styles.get(bl, "-"),
                           linewidth=1.8, alpha=0.85,
                           label=ALGO_LABELS[bl])

        ax.set_xticks(range(1, len(tick_labels) + 1))
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Mean Queue (veh)")
        ax.set_title(scenario)
        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── Console: summary table + % improvement ───────────────────────────────────

def print_table(df: pd.DataFrame, scenario: str):
    if df.empty:
        return
    r_ = rdf(df); b_ = bdf(df)
    cols_show = ["mean_queue", "mean_wait_per_veh", "mean_travel_time",
                 "mean_speed", "throughput"]
    cols_show = [c for c in cols_show if c in df.columns]

    fv = b_[b_["algo"] == "fixed"]["mean_queue"].mean() if "mean_queue" in df.columns else np.nan
    wv = b_[b_["algo"] == "webster"]["mean_queue"].mean() if "mean_queue" in df.columns else np.nan

    def s(sub, col):
        v = sub[col].dropna()
        if v.empty:
            return "N/A"
        return f"{v.mean():6.2f}±{v.std():.2f}" if len(v) > 1 else f"{v.mean():6.2f}"

    def pct(val, ref):
        if np.isnan(ref) or ref == 0 or np.isnan(val):
            return "N/A"
        return f"{(ref - val) / abs(ref) * 100:+.1f}%"

    print(f"\n{'='*95}")
    print(f"  {scenario}")
    print(f"{'='*95}")
    hdr = f"  {'Algo':6s} {'Reward':12s} {'N':>2}  {'Queue±std':>14}  {'Wait/veh±std':>14}  "
    hdr += f"{'TrvlTime':>10}  {'Speed':>7}  {'Thru':>6}  {'vsFixed':>8}  {'vsWebster':>9}"
    print(hdr)
    print("  " + "-" * 90)

    for algo in RL_ALGOS:
        for rw in REWARDS:
            sub = r_[(r_["algo"] == algo) & (r_["reward"] == rw)]
            if sub.empty:
                continue
            q_mean = sub["mean_queue"].mean() if "mean_queue" in sub.columns else np.nan
            row = f"  {algo:6s} {rw:12s} {len(sub):>2}  "
            row += f"{s(sub, 'mean_queue'):>14}  {s(sub, 'mean_wait_per_veh'):>14}  "
            row += f"{s(sub, 'mean_travel_time'):>10}  "
            spd = f"{sub['mean_speed'].mean():.3f}" if "mean_speed" in sub.columns else "N/A"
            thr = f"{sub['throughput'].mean():.0f}" if "throughput" in sub.columns else "N/A"
            row += f"{spd:>7}  {thr:>6}  {pct(q_mean, fv):>8}  {pct(q_mean, wv):>9}"
            print(row)

    print("  " + "-" * 90)
    for bl in BASELINES:
        sub = b_[b_["algo"] == bl]
        if sub.empty:
            continue
        print(f"  {bl:6s} {'—':12s} {len(sub):>2}  "
              f"{s(sub, 'mean_queue'):>14}  {s(sub, 'mean_wait_per_veh'):>14}  "
              f"{s(sub, 'mean_travel_time'):>10}  "
              f"{sub['mean_speed'].mean() if 'mean_speed' in sub.columns else 0:>7.3f}  "
              f"{sub['throughput'].mean() if 'throughput' in sub.columns else 0:>6.0f}")


def save_improvement_csv(df3: pd.DataFrame, df8: pd.DataFrame, out_dir: Path):
    """Luu CSV % improvement (cho reproducibility, khong phai figure)."""
    METRICS = [("mean_queue", "Queue"), ("mean_wait_per_veh", "Wait/veh"),
               ("mean_travel_time", "TravelTime")]
    rows = []
    for scenario, df in [("C3", df3), ("C8", df8)]:
        if df.empty:
            continue
        r_ = rdf(df); b_ = bdf(df)
        for algo in RL_ALGOS:
            for rw in REWARDS:
                sub = r_[(r_["algo"] == algo) & (r_["reward"] == rw)]
                if sub.empty:
                    continue
                row = {"Scenario": scenario, "Algo": algo, "Reward": rw}
                for col, lbl in METRICS:
                    if col not in df.columns:
                        continue
                    m = sub[col].mean()
                    for bname in BASELINES:
                        bv = b_[b_["algo"] == bname][col]
                        bval = bv.mean() if not bv.empty else np.nan
                        pct = round((bval - m) / abs(bval) * 100, 2) if not np.isnan(bval) and bval != 0 else np.nan
                        row[f"vs_{bname}_{lbl}"] = pct
                rows.append(row)

    if rows:
        csv_path = out_dir / "improvement_vs_baselines.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path.name}  (reproducibility)")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--c3",  default="./results_cologne3_real")
    p.add_argument("--c8",  default="./results_cologne8_real")
    p.add_argument("--out", default="./figures")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  plot_cologne_results.py")
    print(f"  C3 dir : {args.c3}")
    print(f"  C8 dir : {args.c8}")
    print(f"  Output : {args.out}/")
    print("=" * 65)

    df3 = load_csv(Path(args.c3) / "all_results.csv")
    df8 = load_csv(Path(args.c8) / "all_results.csv")
    print(f"\nDu lieu: C3={len(df3)} rows | C8={len(df8)} rows")

    print_table(df3, "Cologne3")
    print_table(df8, "Cologne8")

    print("\n=== Generating figures ===")
    fig1a_queue(df3, df8, out_dir / "fig1a_queue.png")
    fig1b_wait( df3, df8, out_dir / "fig1b_wait.png")
    fig2_scalability(df3, df8, out_dir / "fig2_scalability.png")
    fig3_ablation(df3, df8, out_dir / "fig3_ablation.png")
    fig4_boxplot(df3, df8, out_dir / "fig4_boxplot.png")
    save_improvement_csv(df3, df8, out_dir)

    print(f"\nHoan tat. Bieu do luu tai: {out_dir}/")
    for f in sorted(out_dir.glob("fig*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
