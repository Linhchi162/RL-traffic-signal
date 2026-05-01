"""
plot_cologne_results.py — Ve toan bo bieu do thesis cho Cologne3 & Cologne8.

Su dung:
    python plot_cologne_results.py
    python plot_cologne_results.py --c3 results_cologne3_real --c8 results_cologne8_real --out figures

Bieu do tao ra:
    fig1_perf_cologne3.png       — Group 1: Performance bar chart C3
    fig1_perf_cologne8.png       — Group 1: Performance bar chart C8
    fig2_improvement.png         — Group 1: % improvement vs baselines (heatmap)
    fig3_scalability.png         — Group 2: Scalability C3 → C8
    fig4_ablation_cologne3.png   — Group 2: Reward ablation C3
    fig4_ablation_cologne8.png   — Group 2: Reward ablation C8
    fig5_boxplot_cologne3.png    — Group 3: Seed stability C3
    fig5_boxplot_cologne8.png    — Group 3: Seed stability C8
    improvement_vs_baselines.csv — So lieu % cai thien
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
    "random":  "#C62828",
}
ALGO_LABELS  = {"dqn": "DQN", "ddqn": "DDQN", "ppo": "PPO",
                "fixed": "Fixed-time", "webster": "Webster", "random": "Random"}
REWARD_LABELS = {"queue": "Queue", "pressure": "Pressure", "wait-clip": "Wait-clip"}
REWARD_HATCH  = {"queue": "", "pressure": "//", "wait-clip": "xx"}
BASELINES     = ["random", "fixed", "webster"]
RL_ALGOS      = ["dqn", "ddqn", "ppo"]
REWARDS       = ["queue", "pressure", "wait-clip"]

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


# ── I/O helpers ───────────────────────────────────────────────────────────────

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


def rdf(df): return df[~df["algo"].isin(BASELINES)].copy()
def bdf(df): return df[ df["algo"].isin(BASELINES)].copy()


# ── Figure 1: Performance comparison ─────────────────────────────────────────

def fig1_performance(df: pd.DataFrame, scenario: str, out: Path):
    """
    4 subplots (Queue / Wait/veh / Travel Time / Throughput).
    Bars: each (algo x reward) combo + 3 baselines.
    Error bars = std across seeds.
    """
    METRICS = [
        ("mean_queue",        "Mean Queue (veh) ↓",  True),
        ("mean_wait_per_veh", "Mean Wait/veh (s) ↓", True),
        ("mean_travel_time",  "Travel Time (s) ↓",   True),
        ("throughput",        "Throughput (veh) ↑",  False),
    ]
    METRICS = [(c, l, lb) for c, l, lb in METRICS if c in df.columns]
    if not METRICS:
        return

    r = rdf(df); b = bdf(df)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.8 * len(METRICS), 5.5))
    if len(METRICS) == 1:
        axes = [axes]
    fig.suptitle(f"Performance Comparison — {scenario}", fontsize=13,
                 fontweight="bold", y=1.02)

    for ax, (col, ylabel, lower_better) in zip(axes, METRICS):
        groups = []
        for algo in RL_ALGOS:
            for rw in REWARDS:
                sub = r[(r["algo"] == algo) & (r["reward"] == rw)][col].dropna()
                if sub.empty:
                    continue
                lbl = f"{ALGO_LABELS[algo]}\n{REWARD_LABELS[rw]}"
                groups.append((lbl, sub.mean(), sub.std(),
                                ALGO_COLORS[algo], REWARD_HATCH[rw]))
        for bl in BASELINES:
            sub = b[b["algo"] == bl][col].dropna()
            if sub.empty:
                continue
            groups.append((ALGO_LABELS[bl], sub.mean(), 0,
                           ALGO_COLORS[bl], ""))

        if not groups:
            continue

        x = np.arange(len(groups))
        for i, (_, m, s, c, h) in enumerate(groups):
            ax.bar(x[i], m, yerr=s if s > 0 else None,
                   capsize=3, color=c, hatch=h, alpha=0.85,
                   edgecolor="white" if not h else "gray", linewidth=0.5)

        n_rl = len(RL_ALGOS) * len(REWARDS)
        if n_rl < len(groups):
            ax.axvline(n_rl - 0.5, color="black", lw=1.2, ls="--", alpha=0.5)
            ax.text(n_rl - 0.3, ax.get_ylim()[1] * 0.97, "baselines",
                    fontsize=6.5, alpha=0.6, ha="left", va="top")

        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in groups], rotation=40, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split("(")[0].strip())

    # Legend: algo colors + reward hatches
    legend_handles = (
        [plt.Rectangle((0,0),1,1, fc=ALGO_COLORS[a], label=ALGO_LABELS[a])
         for a in RL_ALGOS + BASELINES]
        + [plt.Rectangle((0,0),1,1, fc="white", hatch=REWARD_HATCH[rw], ec="gray",
                         label=f"reward={REWARD_LABELS[rw]}")
           for rw in REWARDS]
    )
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=6, bbox_to_anchor=(0.5, -0.05), framealpha=0.9, fontsize=7.5)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── Figure 2: % Improvement heatmap ──────────────────────────────────────────

def fig2_improvement(df3: pd.DataFrame, df8: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Heatmap % improvement of (algo,reward) vs Fixed-time and Webster."""
    METRICS = [
        ("mean_queue",        "Queue"),
        ("mean_wait_per_veh", "Wait/veh"),
        ("mean_travel_time",  "TravelTime"),
    ]

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
                row = {"Scenario": scenario,
                       "Algo": ALGO_LABELS[algo], "Reward": REWARD_LABELS[rw]}
                for col, lbl in METRICS:
                    if col not in df.columns:
                        continue
                    rl_m = sub[col].mean()
                    for bname in ["fixed", "webster"]:
                        bv = b_[b_["algo"] == bname][col]
                        bval = bv.mean() if not bv.empty else np.nan
                        if np.isnan(bval) or bval == 0:
                            pct = np.nan
                        else:
                            pct = round((bval - rl_m) / abs(bval) * 100, 1)
                        row[f"vs_{bname}_{lbl}"] = pct
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    csv_out = out_dir / "improvement_vs_baselines.csv"
    result.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out.name}")

    # ── Heatmap figure ──
    scenarios_present = [s for s in ["C3", "C8"] if s in result["Scenario"].values]
    if not scenarios_present:
        return result

    metric_cols_f = [f"vs_fixed_{l}"   for _, l in METRICS]
    metric_cols_w = [f"vs_webster_{l}" for _, l in METRICS]
    metric_names  = [l for _, l in METRICS]

    n_panels = len(scenarios_present) * 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * len(scenarios_present), 5.8),
                              gridspec_kw={"wspace": 0.06})
    if n_panels == 1:
        axes = [axes]
    fig.suptitle("% Improvement vs Baselines (green = better than baseline)",
                 fontsize=11, fontweight="bold")

    ax_idx = 0
    for scenario in scenarios_present:
        sub = result[result["Scenario"] == scenario].copy()
        sub["Y"] = sub["Algo"] + " / " + sub["Reward"]

        for title_sfx, cols in [("vs Fixed-time", metric_cols_f),
                                  ("vs Webster",    metric_cols_w)]:
            ax = axes[ax_idx]; ax_idx += 1
            pivot = sub[[c for c in cols if c in sub.columns]].values.astype(float)
            col_names = [m for m, c in zip(metric_names, cols) if c in sub.columns]
            y_labels  = sub["Y"].values

            im = ax.imshow(pivot, aspect="auto", cmap="RdYlGn",
                           vmin=-40, vmax=60, interpolation="nearest")
            ax.set_xticks(range(len(col_names)))
            ax.set_xticklabels(col_names, rotation=25, ha="right", fontsize=8)
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels if ax_idx % 2 == 1 else [], fontsize=7)
            ax.set_title(f"{scenario} {title_sfx}", fontsize=9)

            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    v = pivot[i, j]
                    if not np.isnan(v):
                        tc = "white" if abs(v) > 45 else "black"
                        ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                                fontsize=6.5, color=tc)

    plt.colorbar(im, ax=axes[-1], label="% better than baseline", shrink=0.8)
    fig.tight_layout()
    p = out_dir / "fig2_improvement.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")
    return result


# ── Figure 3: Scalability C3 → C8 ────────────────────────────────────────────

def fig3_scalability(df3: pd.DataFrame, df8: pd.DataFrame, out_dir: Path):
    if df3.empty or df8.empty:
        print("  [Skip] Scalability: missing one scenario")
        return

    METRICS = [
        ("mean_queue",        "Mean Queue (veh) ↓",  True),
        ("mean_wait_per_veh", "Mean Wait/veh (s) ↓", True),
        ("mean_travel_time",  "Travel Time (s) ↓",   True),
        ("throughput",        "Throughput (veh) ↑",  False),
    ]
    METRICS = [(c, l, lb) for c, l, lb in METRICS
               if c in df3.columns and c in df8.columns]

    r3 = rdf(df3); r8 = rdf(df8)
    b3 = bdf(df3); b8 = bdf(df8)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.8 * len(METRICS), 5.5))
    if len(METRICS) == 1:
        axes = [axes]
    fig.suptitle("Scalability: Cologne3 → Cologne8", fontsize=13,
                 fontweight="bold", y=1.02)

    for ax, (col, ylabel, lower_better) in zip(axes, METRICS):
        items = []

        for algo in RL_ALGOS:
            s3 = r3[r3["algo"] == algo]
            s8 = r8[r8["algo"] == algo]
            if s3.empty:
                continue
            grp3 = s3.groupby("reward")[col].mean()
            if grp3.empty:
                continue
            best_r = grp3.idxmin() if lower_better else grp3.idxmax()
            v3 = s3[s3["reward"] == best_r][col].dropna()
            v8 = s8[s8["reward"] == best_r][col].dropna() if not s8.empty else pd.Series(dtype=float)
            if v3.empty:
                continue
            label = f"{ALGO_LABELS[algo]}\n({REWARD_LABELS.get(best_r, best_r)})"
            items.append((label, v3.mean(), v3.std(),
                          v8.mean() if not v8.empty else np.nan,
                          v8.std()  if not v8.empty else 0,
                          ALGO_COLORS[algo]))

        for bl in BASELINES:
            v3 = b3[b3["algo"] == bl][col].dropna()
            v8 = b8[b8["algo"] == bl][col].dropna()
            if v3.empty:
                continue
            items.append((ALGO_LABELS[bl],
                          v3.mean(), 0,
                          v8.mean() if not v8.empty else np.nan, 0,
                          ALGO_COLORS[bl]))

        if not items:
            continue

        n = len(items); x = np.arange(n); w = 0.38
        for i, (lbl, m3, s3_, m8, s8_, c) in enumerate(items):
            ax.bar(x[i] - w/2, m3, w,
                   yerr=s3_ if s3_ > 0 else None,
                   capsize=3, color=c, alpha=0.92,
                   label="Cologne3" if i == 0 else "_")
            if not np.isnan(m8):
                ax.bar(x[i] + w/2, m8, w,
                       yerr=s8_ if s8_ > 0 else None,
                       capsize=3, color=c, alpha=0.40, hatch="//",
                       label="Cologne8" if i == 0 else "_")

        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in items], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split("(")[0].strip())
        if ax is axes[0]:
            ax.legend(["Cologne3", "Cologne8 (hatched)"], fontsize=8)

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    p = out_dir / "fig3_scalability.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Figure 4: Reward function ablation ───────────────────────────────────────

def fig4_ablation(df3: pd.DataFrame, df8: pd.DataFrame, out_dir: Path):
    REWARD_COLORS = {"queue": "#1565C0", "pressure": "#6A1B9A", "wait-clip": "#BF360C"}
    METRICS = [("mean_queue", "Mean Queue (veh) ↓"),
               ("mean_wait_per_veh", "Mean Wait/veh (s) ↓")]

    for scenario, df in [("cologne3", df3), ("cologne8", df8)]:
        if df.empty:
            continue
        r_ = rdf(df)
        valid = [(c, l) for c, l in METRICS if c in df.columns]
        if not valid:
            continue

        fig, axes = plt.subplots(len(valid), len(RL_ALGOS),
                                 figsize=(4.5 * len(RL_ALGOS), 4 * len(valid)),
                                 squeeze=False)
        fig.suptitle(f"Reward Function Ablation — {scenario.capitalize()}",
                     fontsize=12, fontweight="bold", y=1.02)

        for row_i, (col, title) in enumerate(valid):
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
                ax.bar(x, means, yerr=stds, capsize=4, color=clrs, alpha=0.85)
                ax.set_xticks(x); ax.set_xticklabels(lbls)
                if col_i == 0:
                    ax.set_ylabel(title)
                if row_i == 0:
                    ax.set_title(ALGO_LABELS[algo], fontsize=11, fontweight="bold")

                # best marker
                if means:
                    best_i = np.argmin(means)
                    ax.bar(x[best_i], means[best_i], capsize=0,
                           color=clrs[best_i], edgecolor="black",
                           linewidth=2, alpha=0.95)

        fig.tight_layout()
        p = out_dir / f"fig4_ablation_{scenario}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p.name}")


# ── Figure 5: Boxplot seed stability ─────────────────────────────────────────

def fig5_boxplot(df: pd.DataFrame, scenario: str, out_dir: Path):
    r_ = rdf(df)
    if r_.empty:
        return
    METRICS = [(c, l) for c, l in
               [("mean_queue",        "Mean Queue (veh)"),
                ("mean_wait_per_veh", "Mean Wait/veh (s)")]
               if c in df.columns]
    if not METRICS:
        return

    fig, axes = plt.subplots(1, len(METRICS), figsize=(7 * len(METRICS), 5.5))
    if len(METRICS) == 1:
        axes = [axes]
    fig.suptitle(f"Seed Stability (10 seeds) — {scenario}",
                 fontsize=12, fontweight="bold")

    b_ = bdf(df)
    line_styles = {"fixed": "--", "webster": "-.", "random": ":"}

    for ax, (col, ylabel) in zip(axes, METRICS):
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
                           linewidth=1.6, alpha=0.8, label=ALGO_LABELS[bl])

        ax.set_xticks(range(1, len(tick_labels) + 1))
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(ylabel); ax.set_title(ylabel)
        ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    p = out_dir / f"fig5_boxplot_{scenario.lower().replace(' ','_')}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Console summary table ─────────────────────────────────────────────────────

def print_table(df: pd.DataFrame, scenario: str):
    if df.empty:
        return
    print(f"\n{'='*90}")
    print(f"  {scenario}  ({len(df)} rows total)")
    print(f"{'='*90}")
    W = "  {:8s}  {:12s}  {:3s}  {:>14s}  {:>14s}  {:>14s}  {:>7s}"
    print(W.format("Algo", "Reward", "N", "Queue±std", "Wait/veh±std", "TrvlTime±std", "Thru"))
    print("  " + "-" * 80)

    def s(sub, col):
        if col not in sub.columns:
            return "N/A"
        v = sub[col].dropna()
        if v.empty:
            return "N/A"
        return f"{v.mean():6.2f}±{v.std():.2f}" if len(v) > 1 else f"{v.mean():6.2f}"

    r_ = rdf(df); b_ = bdf(df)
    for algo in RL_ALGOS:
        for rw in REWARDS:
            sub = r_[(r_["algo"] == algo) & (r_["reward"] == rw)]
            if sub.empty:
                continue
            thru = f"{sub['throughput'].mean():.0f}" if "throughput" in sub.columns else "N/A"
            print(W.format(algo, rw, str(len(sub)),
                           s(sub, "mean_queue"), s(sub, "mean_wait_per_veh"),
                           s(sub, "mean_travel_time"), thru))
    print("  " + "-" * 80)
    for bl in BASELINES:
        sub = b_[b_["algo"] == bl]
        if sub.empty:
            continue
        thru = f"{sub['throughput'].mean():.0f}" if "throughput" in sub.columns else "N/A"
        print(W.format(bl, "—", str(len(sub)),
                       s(sub, "mean_queue"), s(sub, "mean_wait_per_veh"),
                       s(sub, "mean_travel_time"), thru))


def print_improvement(df: pd.DataFrame, scenario: str):
    if df.empty:
        return
    col = "mean_queue"
    if col not in df.columns:
        return
    b_   = bdf(df)
    fv   = b_[b_["algo"] == "fixed"][col].mean()
    wv   = b_[b_["algo"] == "webster"][col].mean()
    r_   = rdf(df)

    print(f"\n  >> {scenario}: Queue ↓ vs baselines")
    print(f"     {'Algo':8s} {'Reward':12s}  vs Fixed  vs Webster")
    for algo in RL_ALGOS:
        for rw in REWARDS:
            sub = r_[(r_["algo"] == algo) & (r_["reward"] == rw)][col]
            if sub.empty:
                continue
            m = sub.mean()
            pf = (fv - m) / abs(fv) * 100 if fv and not np.isnan(fv) else 0
            pw = (wv - m) / abs(wv) * 100 if wv and not np.isnan(wv) else 0
            print(f"     {algo:8s} {rw:12s}  {pf:+.1f}%    {pw:+.1f}%")


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
    print_improvement(df3, "Cologne3")
    print_improvement(df8, "Cologne8")

    print("\n=== Generating figures ===")

    if not df3.empty:
        fig1_performance(df3, "Cologne3",
                         out_dir / "fig1_perf_cologne3.png")
        fig5_boxplot(df3, "Cologne3", out_dir)

    if not df8.empty:
        fig1_performance(df8, "Cologne8",
                         out_dir / "fig1_perf_cologne8.png")
        fig5_boxplot(df8, "Cologne8", out_dir)

    if not df3.empty or not df8.empty:
        fig2_improvement(df3, df8, out_dir)
        fig4_ablation(df3, df8, out_dir)

    if not df3.empty and not df8.empty:
        fig3_scalability(df3, df8, out_dir)

    print(f"\nHoan tat. Tat ca bieu do luu tai: {out_dir}/")
    for f in sorted(out_dir.glob("fig*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
