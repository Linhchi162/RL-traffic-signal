"""
combine_results.py — Gop ket qua danh gia C3 (real route) va C8 (random route).

Output:
    results_combined/all_results.csv   — all individual seed runs, both scenarios
    results_combined/summary.csv       — mean +- std per (scenario, algo, reward)

Usage:
    python combine_results.py
    python combine_results.py \
        --c3 results_cologne3_real \
        --c8 results_cologne8_rand_curr \
        --out results_combined
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIO_C3 = "C3 (Real Route)"
SCENARIO_C8 = "C8 (Random Route)"

METRICS = ["mean_queue", "mean_wait_per_veh", "mean_speed",
           "throughput", "mean_travel_time"]


def load_all_results(path: Path, scenario: str) -> pd.DataFrame:
    if not path.exists():
        print(f"  [WARN] Not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.insert(0, "scenario", scenario)
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (scenario, algo, reward): mean and std over seeds."""
    rows = []
    for (scenario, algo, reward), grp in df.groupby(["scenario", "algo", "reward"], sort=False):
        row = {"scenario": scenario, "algo": algo, "reward": reward,
               "n_seeds": len(grp)}
        for m in METRICS:
            if m in grp.columns:
                vals = pd.to_numeric(grp[m], errors="coerce").dropna()
                row[f"mean_{m}"] = round(vals.mean(), 4) if len(vals) else np.nan
                row[f"std_{m}"]  = round(vals.std(ddof=1), 4) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--c3",  default="./results_cologne3_real")
    p.add_argument("--c8",  default="./results_cologne8_rand_curr")
    p.add_argument("--out", default="./results_combined")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  combine_results.py")
    print(f"  C3 : {args.c3}/all_results.csv  ({SCENARIO_C3})")
    print(f"  C8 : {args.c8}/all_results.csv  ({SCENARIO_C8})")
    print(f"  Out: {args.out}/")
    print("=" * 60)

    df3 = load_all_results(Path(args.c3) / "all_results.csv", SCENARIO_C3)
    df8 = load_all_results(Path(args.c8) / "all_results.csv", SCENARIO_C8)

    print(f"\n  C3 rows: {len(df3)}  |  C8 rows: {len(df8)}")

    combined = pd.concat([df3, df8], ignore_index=True)
    combined_path = out_dir / "all_results.csv"
    combined.to_csv(combined_path, index=False)
    print(f"  Saved: {combined_path}  ({len(combined)} rows total)")

    summary = build_summary(combined)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}  ({len(summary)} rows)")

    # Print quick overview
    print("\n--- Summary (mean_queue | mean_wait_per_veh | throughput) ---")
    cols = ["scenario", "algo", "reward", "n_seeds"]
    for mc in ["mean_mean_queue", "mean_mean_wait_per_veh", "mean_mean_throughput"]:
        if mc in summary.columns:
            cols.append(mc)
    print(summary[cols].to_string(index=False))

    print("\nHoan tat.")


if __name__ == "__main__":
    main()
