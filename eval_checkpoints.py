"""
eval_checkpoints.py — Vẽ learning curve của PPO bằng cách eval từng checkpoint

Chạy eval nhanh (eval_duration=1800s) trên mỗi checkpoint đã lưu,
xuất CSV + in kết quả để thấy model cải thiện dần hay không.

Sử dụng:
    python eval_checkpoints.py --exp_dir ./experiments/ppo_queue_single_s42
    python eval_checkpoints.py --all_ppo --exp_root ./experiments
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

if "SUMO_HOME" not in os.environ:
    sys.exit("[eval_checkpoints] SUMO_HOME chua duoc khai bao")

_HERE = Path(__file__).parent
_RLTSCQ_NETS = _HERE / "RLTSCQ" / "RLTSCQ-main" / "sumo_rl" / "nets" / "RLQ"
SINGLE_NET   = _RLTSCQ_NETS / "caliberated_net.xml"
_GEN_TEST    = _HERE / "generated_flows" / "test_s999.xml"
SINGLE_ROUTE = _GEN_TEST if _GEN_TEST.exists() else _RLTSCQ_NETS / "test_flows.xml"

EVAL_DURATION = int(os.environ.get("EVAL_DURATION_OVERRIDE", 1800))  # nhanh hơn 7200

from rl_controller.traffic_env import TrafficControlEnv
from rl_controller.state_builder import IntersectionStateExtractor


def _collect_metrics(sumo_conn, signal_ids):
    total_q = total_w = 0.0
    speeds  = []
    for sid in signal_ids:
        try:
            lanes = sumo_conn.trafficlight.getControlledLanes(sid)
            seen  = set()
            for lane in lanes:
                if lane in seen: continue
                seen.add(lane)
                total_q += sumo_conn.lane.getLastStepHaltingNumber(lane)
                total_w += sumo_conn.lane.getWaitingTime(lane)
            for vid in sumo_conn.vehicle.getIDList():
                speeds.append(sumo_conn.vehicle.getSpeed(vid))
        except Exception:
            pass
    return total_q, total_w, float(np.mean(speeds)) if speeds else 0.0


def eval_one_checkpoint(ckpt_path: str, label: str) -> dict:
    from stable_baselines3 import PPO

    try:
        agent = PPO.load(ckpt_path, env=None)
    except Exception as e:
        print(f"  [SKIP] {label}: load loi — {e}")
        return None

    env = TrafficControlEnv(
        net_file=str(SINGLE_NET),
        route_file=str(SINGLE_ROUTE),
        sim_duration=EVAL_DURATION + 300,
        single_agent=True,
        reward_fn="queue",
        obs_class=IntersectionStateExtractor,
        show_warnings=False,
    )
    obs, _ = env.reset()

    queues, waits, speeds = [], [], []
    max_steps = (EVAL_DURATION // 5) + 100

    for _ in range(max_steps):
        sumo_conn = env.sumo
        if sumo_conn is None: break
        if sumo_conn.simulation.getTime() >= EVAL_DURATION: break
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, _, done, _ = env.step(int(action))
        q, w, s = _collect_metrics(sumo_conn, env.signal_ids)
        queues.append(q); waits.append(w); speeds.append(s)
        if done: break

    try: env.close()
    except Exception: pass

    if not queues:
        return None
    return {
        "mean_queue": float(np.mean(queues)),
        "mean_wait":  float(np.mean(waits)),
        "mean_speed": float(np.mean(speeds)),
    }


def eval_experiment(exp_dir: Path, save_csv: bool = True):
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.exists():
        print(f"  [SKIP] Khong co thu muc checkpoints: {exp_dir.name}")
        return []

    ckpt_files = sorted(ckpt_dir.glob("ppo_ckpt_*_steps.zip"))
    if not ckpt_files:
        print(f"  [SKIP] Khong co checkpoint: {exp_dir.name}")
        return []

    # Them final model neu co
    final = exp_dir / "ppo_final_model.zip"
    extra = []
    if final.exists():
        extra = [final]

    print(f"\n>>> {exp_dir.name}  ({len(ckpt_files)} checkpoints + {'final' if extra else 'no final'})")
    print(f"    {'Step':>10}  {'Queue':>8}  {'Wait':>10}  {'Speed':>7}  {'vs Random':>10}")
    print(f"    {'-'*10}  {'-'*8}  {'-'*10}  {'-'*7}  {'-'*10}")

    RANDOM_QUEUE = 63.79   # from eval baseline
    results = []

    for ckpt in ckpt_files + extra:
        # parse step from filename
        stem = ckpt.stem
        if stem == "ppo_final_model":
            step = -1   # marker for final
        else:
            try:
                step = int(stem.split("_steps")[0].split("_")[-1])
            except ValueError:
                step = -1

        label = f"{exp_dir.name}@{step}"
        m = eval_one_checkpoint(str(ckpt), label)
        if m is None:
            continue

        pct = (m["mean_queue"] - RANDOM_QUEUE) / RANDOM_QUEUE * 100
        step_str = f"{step:,}" if step >= 0 else "final"
        sign = "✓" if m["mean_queue"] < RANDOM_QUEUE else "✗"
        print(f"    {step_str:>10}  {m['mean_queue']:8.2f}  "
              f"{m['mean_wait']:10.1f}  {m['mean_speed']:7.3f}  "
              f"{pct:+9.1f}% {sign}")
        results.append({"step": step if step >= 0 else 999999, **m})

    if save_csv and results:
        out = exp_dir / "checkpoint_learning_curve.csv"
        fields = ["step", "mean_queue", "mean_wait", "mean_speed"]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results)
        print(f"    → Luu: {out}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir",  type=str, default=None,
                    help="Thu muc 1 experiment (vd: experiments/ppo_queue_single_s42)")
    ap.add_argument("--all_ppo",  action="store_true",
                    help="Chay tat ca experiment PPO trong exp_root")
    ap.add_argument("--exp_root", default="./experiments")
    ap.add_argument("--eval_duration", type=int, default=1800,
                    help="Do dai eval (giay). Mac dinh 1800 de nhanh")
    ap.add_argument("--only",     type=str, default=None,
                    help="Chi chay cac exp khop ten (phan cach bang dau phay)")
    args = ap.parse_args()

    global EVAL_DURATION
    EVAL_DURATION = args.eval_duration

    t0 = time.time()

    if args.exp_dir:
        eval_experiment(Path(args.exp_dir))
    elif args.all_ppo:
        root = Path(args.exp_root)
        only = [x.strip() for x in args.only.split(",")] if args.only else None
        exps = sorted([d for d in root.iterdir()
                       if d.is_dir()
                       and d.name.startswith("ppo_")
                       and not any(d.name.endswith(s) for s in ("_failed", "_bak", "_old"))
                       and (only is None or any(o in d.name for o in only))])
        print(f"Tim thay {len(exps)} PPO experiments")
        for exp in exps:
            eval_experiment(exp)
    else:
        ap.print_help()
        sys.exit(0)

    print(f"\nHoan tat trong {(time.time()-t0)/60:.1f} phut")


if __name__ == "__main__":
    main()
