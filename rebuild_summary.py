"""Tai sinh summary.csv tu all_results.csv (them median, khong can chay SUMO lai)."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
src = results_dir / "all_results.csv"
dst = results_dir / "summary.csv"

if not src.exists():
    sys.exit(f"Khong tim thay {src}")

rows = list(csv.DictReader(open(src)))
grouped = defaultdict(list)
for r in rows:
    key = (r["algo"], r["reward"], r.get("obs_mode", "-"), r.get("scope", "multi"), r["flow"])
    grouped[key].append(r)

fields = ["algo", "reward", "obs_mode", "scope", "flow", "n_runs",
          "mean_queue", "std_queue", "median_queue",
          "mean_wait",  "std_wait",  "median_wait",
          "mean_speed", "std_speed", "median_speed"]

out = []
for (algo, reward, obs_mode, scope, flow), rs in sorted(grouped.items()):
    q = [float(r["mean_queue"]) for r in rs]
    w = [float(r["mean_wait"])  for r in rs]
    s = [float(r["mean_speed"]) for r in rs]
    out.append({
        "algo": algo, "reward": reward, "obs_mode": obs_mode,
        "scope": scope, "flow": flow, "n_runs": len(rs),
        "mean_queue":   round(np.mean(q),   3),
        "std_queue":    round(np.std(q),    3),
        "median_queue": round(np.median(q), 3),
        "mean_wait":    round(np.mean(w),   2),
        "std_wait":     round(np.std(w),    2),
        "median_wait":  round(np.median(w), 2),
        "mean_speed":   round(np.mean(s),   4),
        "std_speed":    round(np.std(s),    4),
        "median_speed": round(np.median(s), 4),
    })

with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out)

print(f"Ghi {len(out)} rows -> {dst}")
print("\nMedian queue (high flow):")
for row in out:
    if row["flow"] == "high":
        print(f"  {row['algo']:8} {row['reward']:20} {row['obs_mode']:12} "
              f"median={row['median_queue']:6}  mean={row['mean_queue']:6}  std={row['std_queue']}")
