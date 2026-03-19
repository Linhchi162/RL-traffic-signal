"""
analyze_results.py — Phân tích thống kê kết quả đánh giá

Đọc all_results.csv, thực hiện:
  1. Bảng mean ± std queue/wait theo (algo, reward)
  2. T-test: mỗi nhóm vs random policy baseline
  3. T-test: PPO vs DQN cho cùng reward type
  4. Phân loại: model nào thực sự học được

Sử dụng:
    python analyze_results.py --results_dir ./results_s999
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")


RANDOM_QUEUE = None   # được đọc từ CSV


def load_results(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _vals(rows, col):
    out = []
    for r in rows:
        try:
            v = float(r.get(col, "") or "")
            out.append(v)
        except ValueError:
            pass
    return out


def _group(rows):
    """Group rows theo (algo, reward), bỏ qua baselines (fixed/webster/random)."""
    g = defaultdict(list)
    for r in rows:
        algo   = r.get("algo", "")
        reward = r.get("reward", "-")
        if algo in ("fixed", "webster", "random"):
            continue
        g[(algo, reward)].append(r)
    return g


def _ttest_vs_random(vals, random_mean):
    """One-sample t-test: H0 = mean(vals) == random_mean."""
    if len(vals) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_1samp(vals, random_mean)
    return t, p


def _ttest_two(a_vals, b_vals):
    """Independent two-sample t-test."""
    if len(a_vals) < 2 or len(b_vals) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a_vals, b_vals, equal_var=False)
    return t, p


def _stars(p):
    if np.isnan(p): return "  "
    if p < 0.001:   return "***"
    if p < 0.01:    return "** "
    if p < 0.05:    return "*  "
    return "   "


def _pct_change(val, ref):
    return (val - ref) / ref * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="./results_s999")
    ap.add_argument("--metric",      default="mean_queue",
                    choices=["mean_queue", "mean_wait", "mean_speed"])
    args = ap.parse_args()

    csv_path = Path(args.results_dir) / "all_results.csv"
    if not csv_path.exists():
        sys.exit(f"Khong tim thay {csv_path}")

    rows = load_results(csv_path)
    metric = args.metric
    higher_is_better = (metric == "mean_speed")

    # Tách baseline
    baselines = {r["algo"]: _vals([r], metric) for r in rows
                 if r.get("algo") in ("fixed", "webster", "random")}
    random_vals = baselines.get("random", [])
    random_mean = float(np.mean(random_vals)) if random_vals else None
    fixed_mean  = float(np.mean(baselines.get("fixed",   [0])))
    webster_mean = float(np.mean(baselines.get("webster", [0])))

    print(f"\n{'='*65}")
    print(f"  Phan tich: {metric}  |  dir: {args.results_dir}")
    print(f"{'='*65}")
    print(f"  Baselines:")
    print(f"    Random  : {random_mean:.2f}" if random_mean else "    Random  : N/A")
    print(f"    Fixed   : {fixed_mean:.2f}")
    print(f"    Webster : {webster_mean:.2f}")
    print()

    grouped = _group(rows)

    # -----------------------------------------------------------------------
    # Bảng 1: mean ± std và t-test vs random
    # -----------------------------------------------------------------------
    print(f"  {'Algo':8s}  {'Reward':12s}  {'N':>2}  {'Mean':>8}  {'Std':>7}  "
          f"{'vs Random':>10}  {'p-val':>8}  {'Sig':3}  {'Beat Random?':>12}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*2}  {'-'*8}  {'-'*7}  "
          f"{'-'*10}  {'-'*8}  {'-'*3}  {'-'*12}")

    beats_random = []
    fails_random = []

    for (algo, reward), group_rows in sorted(grouped.items()):
        vals = _vals(group_rows, metric)
        if not vals:
            continue
        mean_v = float(np.mean(vals))
        std_v  = float(np.std(vals))
        n      = len(vals)

        if random_mean is not None:
            t, p = _ttest_vs_random(vals, random_mean)
            pct  = _pct_change(mean_v, random_mean)
            sig  = _stars(p)
            # "tốt hơn" = queue thấp hơn hoặc speed cao hơn
            if higher_is_better:
                better = mean_v > random_mean
            else:
                better = mean_v < random_mean
            beat_str = "YES ✓" if better else "NO  ✗"
            if better:
                beats_random.append((algo, reward, mean_v))
            else:
                fails_random.append((algo, reward, mean_v))
            p_str = f"{p:.4f}" if not np.isnan(p) else "  N/A "
            print(f"  {algo:8s}  {reward:12s}  {n:>2}  {mean_v:8.2f}  "
                  f"{std_v:7.2f}  {pct:+9.1f}%  {p_str:>8}  {sig:3}  {beat_str:>12}")
        else:
            print(f"  {algo:8s}  {reward:12s}  {n:>2}  {mean_v:8.2f}  {std_v:7.2f}")

    print(f"\n  (* p<0.05  ** p<0.01  *** p<0.001)")

    # -----------------------------------------------------------------------
    # Bảng 2: PPO vs DQN/DDQN cùng reward
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("  So sanh PPO vs DQN / DDQN (cung reward type)")
    print(f"{'='*65}")
    print(f"  {'Reward':12s}  {'PPO mean':>9}  {'DQN mean':>9}  {'DDQN mean':>10}  "
          f"{'PPO<DQN?':>9}  {'PPO<DDQN?':>10}")
    print(f"  {'-'*12}  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*9}  {'-'*10}")

    rewards = set(r for _, r in grouped.keys())
    for reward in sorted(rewards):
        ppo_vals  = _vals(grouped.get(("ppo",  reward), []), metric)
        dqn_vals  = _vals(grouped.get(("dqn",  reward), []), metric)
        ddqn_vals = _vals(grouped.get(("ddqn", reward), []), metric)

        ppo_m  = float(np.mean(ppo_vals))  if ppo_vals  else float("nan")
        dqn_m  = float(np.mean(dqn_vals))  if dqn_vals  else float("nan")
        ddqn_m = float(np.mean(ddqn_vals)) if ddqn_vals else float("nan")

        _, p_dqn  = _ttest_two(ppo_vals, dqn_vals)
        _, p_ddqn = _ttest_two(ppo_vals, ddqn_vals)

        def _cmp(ppo, other, p):
            if np.isnan(ppo) or np.isnan(other): return "   N/A"
            if higher_is_better:
                sign = ">" if ppo > other else "<"
            else:
                sign = "<" if ppo < other else ">"
            return f"{sign} {_stars(p)}"

        print(f"  {reward:12s}  {ppo_m:9.2f}  {dqn_m:9.2f}  {ddqn_m:10.2f}  "
              f"{_cmp(ppo_m, dqn_m, p_dqn):>9}  {_cmp(ppo_m, ddqn_m, p_ddqn):>10}")

    # -----------------------------------------------------------------------
    # Bảng 3: Per-seed breakdown — model nào thực sự học được
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("  Chi tiet per-seed: model nao THUC SU hoc duoc")
    print(f"  (so sanh voi random={random_mean:.2f})")
    print(f"{'='*65}")

    for (algo, reward), group_rows in sorted(grouped.items()):
        seeds_ok  = []
        seeds_bad = []
        for r in group_rows:
            v = _vals([r], metric)
            if not v: continue
            val  = v[0]
            seed = r.get("seed", "?")
            if higher_is_better:
                ok = val > random_mean
            else:
                ok = val < random_mean
            if ok:
                seeds_ok.append(f"s{seed}={val:.1f}")
            else:
                seeds_bad.append(f"s{seed}={val:.1f}")
        ok_str  = ", ".join(seeds_ok)  or "—"
        bad_str = ", ".join(seeds_bad) or "—"
        n_ok = len(seeds_ok)
        n    = len(group_rows)
        bar  = "✓" * n_ok + "✗" * (n - n_ok)
        print(f"  {algo:6s}/{reward:12s}  [{bar}]  OK: {ok_str}")
        if seeds_bad:
            print(f"  {' ':21s}         FAIL: {bad_str}")

    # -----------------------------------------------------------------------
    # Tổng kết
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("  TONG KET")
    print(f"{'='*65}")
    algo_summary = defaultdict(lambda: {"ok": 0, "total": 0})
    for (algo, reward), group_rows in grouped.items():
        for r in group_rows:
            v = _vals([r], metric)
            if not v: continue
            algo_summary[algo]["total"] += 1
            val = v[0]
            if higher_is_better:
                ok = val > random_mean
            else:
                ok = val < random_mean
            if ok:
                algo_summary[algo]["ok"] += 1

    for algo, s in sorted(algo_summary.items()):
        pct = s["ok"] / s["total"] * 100 if s["total"] else 0
        print(f"  {algo:8s}: {s['ok']:2d}/{s['total']:2d} seeds beat random ({pct:.0f}%)")


if __name__ == "__main__":
    main()
