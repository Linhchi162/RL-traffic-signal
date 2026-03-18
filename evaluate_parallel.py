"""
evaluate_parallel.py — Chay evaluate song song tren nhieu process

Moi process chay 1 subset models, ket qua ghi vao thu muc tam,
sau do merge thanh 1 CSV duy nhat.

Su dung:
    python evaluate_parallel.py --jobs 4 --scope multi --save_dir ./results_multi
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def _rebuild_summary(all_rows: list, save_dir: Path):
    grouped = defaultdict(list)
    for row in all_rows:
        key = (row["algo"], row["reward"], row.get("obs_mode", "-"),
               row.get("scope", "multi"), row["flow"])
        grouped[key].append(row)

    def _f(rows, col, default=0.0):
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(col, default) or default))
            except ValueError:
                pass
        return vals

    summary_rows = []
    for (algo, reward, obs_mode, scope, flow), rows in sorted(grouped.items()):
        queues = _f(rows, "mean_queue")
        waits  = _f(rows, "mean_wait")
        wpv    = _f(rows, "mean_wait_per_veh")
        speeds = _f(rows, "mean_speed")
        thru   = _f(rows, "throughput")
        ttime  = _f(rows, "mean_travel_time")

        def _stat(vals, dp):
            if not vals:
                return 0, 0, 0
            return round(float(np.mean(vals)), dp), round(float(np.std(vals)), dp), round(float(np.median(vals)), dp)

        mq, sq, medq = _stat(queues, 3)
        mw, sw, _    = _stat(waits,  2)
        mwpv, _, _   = _stat(wpv,    2)
        ms, ss, _    = _stat(speeds,  4)
        mt, st, _    = _stat(thru,   1)
        mtt, stt, _  = _stat(ttime,  2)

        summary_rows.append({
            "algo": algo, "reward": reward, "obs_mode": obs_mode,
            "scope": scope, "flow": flow, "n_runs": len(rows),
            "mean_queue": mq, "std_queue": sq, "median_queue": medq,
            "mean_wait": mw, "std_wait": sw,
            "mean_wait_per_veh": mwpv,
            "mean_speed": ms, "std_speed": ss,
            "mean_throughput": mt, "std_throughput": st,
            "mean_travel_time": mtt, "std_travel_time": stt,
        })

    sum_fields = [
        "algo", "reward", "obs_mode", "scope", "flow", "n_runs",
        "mean_queue", "std_queue", "median_queue",
        "mean_wait", "std_wait", "mean_wait_per_veh",
        "mean_speed", "std_speed",
        "mean_throughput", "std_throughput",
        "mean_travel_time", "std_travel_time",
    ]
    with open(save_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)
    print(f"  summary.csv: {len(summary_rows)} rows")


def main():
    ap = argparse.ArgumentParser(
        description="Parallel evaluate — chay nhieu process cung luc"
    )
    ap.add_argument("--jobs",       type=int, default=4,
                    help="So process song song (khuyen nghi: so vCPU / 2)")
    ap.add_argument("--scope",      default="multi", choices=["single", "multi"])
    ap.add_argument("--save_dir",   default="./results_multi")
    ap.add_argument("--models_dir", default="./experiments")
    ap.add_argument("--flows",      nargs="+", default=["high", "medium", "low"])
    ap.add_argument("--eval_duration", type=int, default=3000,
                    help="Do dai mo phong (giay) — mac dinh 3000 de nhanh hon")
    ap.add_argument("--skip_ae",    action="store_true", default=True)
    args = ap.parse_args()

    save_dir   = Path(args.save_dir)
    models_dir = Path(args.models_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Lay danh sach models tu evaluate_all.py
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_all import discover_models
    models = discover_models(models_dir, skip_ae=args.skip_ae)
    if not models:
        print("Khong tim thay model nao trong", models_dir)
        return

    model_names = [Path(m[3]).parent.name for m in models]
    n_jobs = min(args.jobs, len(model_names))

    # Chia models thanh n_jobs nhom
    chunks = [model_names[i::n_jobs] for i in range(n_jobs)]

    print(f"Eval {len(model_names)} models | {n_jobs} jobs | eval_duration={args.eval_duration}s")
    print(f"Scope: {args.scope} | Flows: {args.flows}")
    print("=" * 55)

    temp_dirs = [Path(tempfile.mkdtemp(prefix="eval_tmp_")) for _ in range(n_jobs)]
    procs = []

    python = sys.executable

    for i, (chunk, tmp_dir) in enumerate(zip(chunks, temp_dirs)):
        if not chunk:
            continue

        cmd = [
            python, "evaluate_all.py",
            "--scope",      args.scope,
            "--save_dir",   str(tmp_dir),
            "--models_dir", str(models_dir),
            "--only",       ",".join(chunk),
            "--skip_ae",
        ]
        # Chi job 0 chay baselines (fixed + webster)
        if i > 0:
            cmd += ["--skip_fixed", "--skip_webster"]

        if args.flows != ["high", "medium", "low"]:
            cmd += ["--flows"] + args.flows

        # Ghi de EVAL_DURATION qua env var
        env = os.environ.copy()
        env["EVAL_DURATION_OVERRIDE"] = str(args.eval_duration)

        log_path = save_dir / f"eval_job{i}.log"
        log_file = open(log_path, "w", encoding="utf-8")

        proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, env=env)
        procs.append((proc, log_file, tmp_dir, i, len(chunk)))
        print(f"  [Job {i}] {len(chunk)} models → log: {log_path.name}")

    print("\nDang chay... (theo doi: tail -f eval_job0.log)")

    # Cho tat ca process hoan thanh
    for proc, log_file, tmp_dir, idx, n_m in procs:
        proc.wait()
        log_file.close()
        status = "OK" if proc.returncode == 0 else f"EXIT={proc.returncode}"
        print(f"  [Job {idx}] {status} ({n_m} models)")

    print("\nMerge ket qua...")

    # Merge all_results.csv
    all_rows = []
    ts_rows  = []
    seen_baselines = set()

    for _, _, tmp_dir, _, _ in procs:
        all_csv = tmp_dir / "all_results.csv"
        ts_csv  = tmp_dir / "timeseries_results.csv"

        if all_csv.exists():
            with open(all_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    # Tranh trung baseline (fixed/webster) tu nhieu job
                    if row["algo"] in ("fixed", "webster"):
                        bkey = (row["algo"], row.get("flow",""), row.get("scope",""))
                        if bkey in seen_baselines:
                            continue
                        seen_baselines.add(bkey)
                    all_rows.append(row)

        if ts_csv.exists():
            with open(ts_csv, newline="", encoding="utf-8") as f:
                ts_rows.extend(list(csv.DictReader(f)))

    # Ghi merged CSVs
    if all_rows:
        fields = list(all_rows[0].keys())
        with open(save_dir / "all_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        print(f"  all_results.csv: {len(all_rows)} rows")

    if ts_rows:
        fields = list(ts_rows[0].keys())
        with open(save_dir / "timeseries_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(ts_rows)
        print(f"  timeseries_results.csv: {len(ts_rows)} rows")

    # Rebuild summary
    if all_rows:
        _rebuild_summary(all_rows, save_dir)

    # Xoa thu muc tam
    for _, _, tmp_dir, _, _ in procs:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'='*55}")
    print(f"  HOAN TAT — {len(all_rows)} records trong {save_dir}")
    print(f"{'='*55}")
    print(f"\nBuoc tiep theo:")
    print(f"  python plot_multi_figures.py --results_dir {save_dir} --scope {args.scope} --all")


if __name__ == "__main__":
    main()
