"""
make_results_table.py — Tao bang so lieu chinh cho thesis.

Hang: Fixed-time, Webster, roi DQN / DDQN / PPO x 3 reward.
Cot : Algorithm | Reward | C3 Queue | C8 Queue |
                           C3 Wait/veh | C8 Wait/veh |
                           C3 Throughput | C8 Throughput
Tat ca mean +- std tren 10 seed.

Output:
    figures/results_table.tex  — LaTeX booktabs (dan vao thesis)
    figures/results_table.csv  — CSV de kiem tra
    (console print de doc nhanh)

Su dung:
    python make_results_table.py
    python make_results_table.py \\
        --c3 results_cologne3_real --c8 results_cologne8_real --out figures
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

BASELINES = ["fixed", "webster"]
RL_ALGOS  = ["dqn", "ddqn", "ppo"]
REWARDS   = ["queue", "pressure", "wait-clip"]

ALGO_LABEL   = {"fixed": "Fixed-time", "webster": "Webster",
                "dqn": "DQN", "ddqn": "DDQN", "ppo": "PPO"}
REWARD_LABEL = {"queue": "Queue", "pressure": "Pressure",
                "wait-clip": "Wait-clip", "—": "—"}


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"  [WARN] Khong tim thay: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str)
    for col in ["mean_queue", "mean_wait_per_veh", "throughput",
                "mean_travel_time", "mean_speed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Stat extraction ───────────────────────────────────────────────────────────

def get_stats(df: pd.DataFrame, algo: str, reward: str | None):
    """Returns (queue_m, queue_s, wait_m, wait_s, thru_m, thru_s) or Nones."""
    if df.empty:
        return (np.nan,) * 6
    if algo in BASELINES:
        sub = df[df["algo"] == algo]
    else:
        sub = df[(df["algo"] == algo) & (df["reward"] == reward)]
    if sub.empty:
        return (np.nan,) * 6
    q  = sub["mean_queue"].dropna()
    w  = sub["mean_wait_per_veh"].dropna()
    th = sub["throughput"].dropna()
    return (q.mean(),  q.std(ddof=1) if len(q) > 1 else 0.0,
            w.mean(),  w.std(ddof=1) if len(w) > 1 else 0.0,
            th.mean(), th.std(ddof=1) if len(th) > 1 else 0.0)


# ── Formatting ────────────────────────────────────────────────────────────────

def _fmt(mean: float, std: float, dec: int, bold: bool = False) -> str:
    if np.isnan(mean):
        return "—"
    std_part = f" $\\pm$ {std:.{dec}f}" if std > 0 else ""
    val = f"{mean:.{dec}f}{std_part}"
    return f"\\textbf{{{val}}}" if bold else val


def _fmt_plain(mean: float, std: float, dec: int) -> str:
    if np.isnan(mean):
        return "—"
    std_part = f" ± {std:.{dec}f}" if std > 0 else ""
    return f"{mean:.{dec}f}{std_part}"


# ── Build table rows ──────────────────────────────────────────────────────────

def build_rows(df3: pd.DataFrame, df8: pd.DataFrame) -> list[dict]:
    rows = []

    # Baselines first
    for algo in BASELINES:
        s3 = get_stats(df3, algo, None)
        s8 = get_stats(df8, algo, None)
        rows.append({"algo": algo, "reward": "—",
                     "c3": s3, "c8": s8})

    # RL combos
    for algo in RL_ALGOS:
        for rw in REWARDS:
            s3 = get_stats(df3, algo, rw)
            s8 = get_stats(df8, algo, rw)
            rows.append({"algo": algo, "reward": rw,
                         "c3": s3, "c8": s8})
    return rows


# ── Find best RL per metric (for bolding) ────────────────────────────────────

def best_rl(rows: list[dict]):
    """Returns indices of best (min) queue, wait, best (max) throughput for C3/C8."""
    rl_rows  = [i for i, r in enumerate(rows) if r["algo"] not in BASELINES]
    best = {}
    for scen in ("c3", "c8"):
        vals_q  = [(rows[i][scen][0], i) for i in rl_rows if not np.isnan(rows[i][scen][0])]
        vals_w  = [(rows[i][scen][2], i) for i in rl_rows if not np.isnan(rows[i][scen][2])]
        vals_th = [(rows[i][scen][4], i) for i in rl_rows if not np.isnan(rows[i][scen][4])]
        best[f"{scen}_q"]  = min(vals_q,  key=lambda x: x[0])[1] if vals_q  else -1
        best[f"{scen}_w"]  = min(vals_w,  key=lambda x: x[0])[1] if vals_w  else -1
        best[f"{scen}_th"] = max(vals_th, key=lambda x: x[0])[1] if vals_th else -1
    return best


# ── LaTeX output ──────────────────────────────────────────────────────────────

LATEX_HEADER = r"""\begin{table}[htbp]
\centering
\caption{Performance comparison on Cologne3 and Cologne8 (mean $\pm$ std over 10 seeds).
         Queue = mean halting vehicles per junction step;
         Wait/veh = mean waiting time per vehicle (s);
         Throughput = total vehicles departed.
         Bold = best RL result per metric.}
\label{tab:results_main}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccc}
\toprule
\multirow{2}{*}{\textbf{Algorithm}} & \multirow{2}{*}{\textbf{Reward}}
  & \multicolumn{2}{c}{\textbf{Queue} $\downarrow$}
  & \multicolumn{2}{c}{\textbf{Wait/veh (s)} $\downarrow$}
  & \multicolumn{2}{c}{\textbf{Throughput} $\uparrow$} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
 & & \textbf{C3} & \textbf{C8} & \textbf{C3} & \textbf{C8} & \textbf{C3} & \textbf{C8} \\
\midrule
"""

LATEX_FOOTER = r"""\bottomrule
\end{tabular}}
\end{table}
"""


def render_latex(rows: list[dict], best: dict) -> str:
    lines = [LATEX_HEADER]
    prev_algo = None

    for i, row in enumerate(rows):
        algo   = row["algo"]
        reward = row["reward"]
        s3, s8 = row["c3"], row["c8"]

        # Separator between baselines and RL
        if prev_algo in BASELINES and algo not in BASELINES:
            lines.append("\\midrule\n")
        # Separator between algo groups
        elif prev_algo not in BASELINES and algo not in BASELINES and algo != prev_algo:
            lines.append("\\addlinespace[3pt]\n")

        prev_algo = algo

        # Bold flags
        b3q  = best.get("c3_q")  == i
        b3w  = best.get("c3_w")  == i
        b3th = best.get("c3_th") == i
        b8q  = best.get("c8_q")  == i
        b8w  = best.get("c8_w")  == i
        b8th = best.get("c8_th") == i

        albl = ALGO_LABEL[algo]
        rlbl = REWARD_LABEL[reward]

        c3q  = _fmt(s3[0], s3[1], 2, b3q)
        c8q  = _fmt(s8[0], s8[1], 2, b8q)
        c3w  = _fmt(s3[2], s3[3], 1, b3w)
        c8w  = _fmt(s8[2], s8[3], 1, b8w)
        c3th = _fmt(s3[4], s3[5], 0, b3th)
        c8th = _fmt(s8[4], s8[5], 0, b8th)

        lines.append(
            f"{albl} & {rlbl} & {c3q} & {c8q} & {c3w} & {c8w} & {c3th} & {c8th} \\\\\n"
        )

    lines.append(LATEX_FOOTER)
    return "".join(lines)


# ── Console print ─────────────────────────────────────────────────────────────

def print_console(rows: list[dict], best: dict):
    # Column widths
    W = [14, 10, 16, 16, 16, 16, 14, 14]
    headers = ["Algorithm", "Reward",
               "C3 Queue", "C8 Queue",
               "C3 Wait/veh", "C8 Wait/veh",
               "C3 Thru", "C8 Thru"]

    sep = "+" + "+".join("-" * w for w in W) + "+"
    hdr = "|" + "|".join(h.center(W[j]) for j, h in enumerate(headers)) + "|"
    print(sep); print(hdr); print(sep)

    prev_algo = None
    for i, row in enumerate(rows):
        algo   = row["algo"]
        reward = row["reward"]
        s3, s8 = row["c3"], row["c8"]

        if prev_algo in BASELINES and algo not in BASELINES:
            print(sep)
        elif prev_algo not in BASELINES and algo not in BASELINES and algo != prev_algo:
            print("|" + " " * (sum(W) + len(W) - 1) + "|")

        prev_algo = algo

        cols = [
            ALGO_LABEL[algo],
            REWARD_LABEL[reward],
            _fmt_plain(s3[0], s3[1], 2),
            _fmt_plain(s8[0], s8[1], 2),
            _fmt_plain(s3[2], s3[3], 1),
            _fmt_plain(s8[2], s8[3], 1),
            _fmt_plain(s3[4], s3[5], 0),
            _fmt_plain(s8[4], s8[5], 0),
        ]
        # Mark best
        marks = ["", "",
                 "*" if best.get("c3_q") == i else "",
                 "*" if best.get("c8_q") == i else "",
                 "*" if best.get("c3_w") == i else "",
                 "*" if best.get("c8_w") == i else "",
                 "*" if best.get("c3_th") == i else "",
                 "*" if best.get("c8_th") == i else ""]
        row_cells = [f"{c}{m}".center(W[j])
                     for j, (c, m) in enumerate(zip(cols, marks))]
        print("|" + "|".join(row_cells) + "|")

    print(sep)
    print("  * = best RL result for that metric")


# ── CSV export ────────────────────────────────────────────────────────────────

def save_csv(rows: list[dict], path: Path):
    records = []
    for row in rows:
        s3, s8 = row["c3"], row["c8"]
        records.append({
            "algorithm":      ALGO_LABEL[row["algo"]],
            "reward":         REWARD_LABEL[row["reward"]],
            "c3_queue_mean":  round(s3[0], 4) if not np.isnan(s3[0]) else "",
            "c3_queue_std":   round(s3[1], 4) if not np.isnan(s3[1]) else "",
            "c8_queue_mean":  round(s8[0], 4) if not np.isnan(s8[0]) else "",
            "c8_queue_std":   round(s8[1], 4) if not np.isnan(s8[1]) else "",
            "c3_wait_mean":   round(s3[2], 2) if not np.isnan(s3[2]) else "",
            "c3_wait_std":    round(s3[3], 2) if not np.isnan(s3[3]) else "",
            "c8_wait_mean":   round(s8[2], 2) if not np.isnan(s8[2]) else "",
            "c8_wait_std":    round(s8[3], 2) if not np.isnan(s8[3]) else "",
            "c3_thru_mean":   round(s3[4], 1) if not np.isnan(s3[4]) else "",
            "c3_thru_std":    round(s3[5], 1) if not np.isnan(s3[5]) else "",
            "c8_thru_mean":   round(s8[4], 1) if not np.isnan(s8[4]) else "",
            "c8_thru_std":    round(s8[5], 1) if not np.isnan(s8[5]) else "",
        })
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"  Saved: {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

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
    print("  make_results_table.py")
    print(f"  C3 : {args.c3}/all_results.csv")
    print(f"  C8 : {args.c8}/all_results.csv")
    print(f"  Out: {args.out}/")
    print("=" * 65)

    df3 = load_csv(Path(args.c3) / "all_results.csv")
    df8 = load_csv(Path(args.c8) / "all_results.csv")
    print(f"  C3: {len(df3)} rows | C8: {len(df8)} rows\n")

    rows = build_rows(df3, df8)
    best = best_rl(rows)

    print_console(rows, best)

    tex = render_latex(rows, best)
    tex_path = out_dir / "results_table.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"\n  Saved: results_table.tex")

    save_csv(rows, out_dir / "results_table.csv")
    print("\nHoan tat.")


if __name__ == "__main__":
    main()
