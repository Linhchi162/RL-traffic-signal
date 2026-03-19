"""
eval_best_checkpoint.py — Tìm và eval checkpoint tốt nhất của mỗi PPO model

Đọc checkpoint_learning_curve.csv (từ eval_checkpoints.py),
tìm checkpoint có mean_queue thấp nhất, eval lại với full 7200s.

Sử dụng:
    python eval_best_checkpoint.py --exp_root ./experiments --save_dir ./results_best_ckpt
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

if "SUMO_HOME" not in os.environ:
    sys.exit("[eval_best_checkpoint] SUMO_HOME chua duoc khai bao")

_HERE = Path(__file__).parent
_RLTSCQ_NETS = _HERE / "RLTSCQ" / "RLTSCQ-main" / "sumo_rl" / "nets" / "RLQ"
SINGLE_NET   = _RLTSCQ_NETS / "caliberated_net.xml"
_GEN_TEST    = _HERE / "generated_flows" / "test_s999.xml"
SINGLE_ROUTE = _GEN_TEST if _GEN_TEST.exists() else _RLTSCQ_NETS / "test_flows.xml"
EVAL_DURATION = int(os.environ.get("EVAL_DURATION_OVERRIDE", 7_200))

from rl_controller.traffic_env import TrafficControlEnv
from rl_controller.state_builder import IntersectionStateExtractor


def _collect_step(sumo_conn, signal_ids):
    total_q = total_w = 0.0
    speeds = []
    for sid in signal_ids:
        try:
            lanes = sumo_conn.trafficlight.getControlledLanes(sid)
            seen = set()
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


def eval_model(model_path: str) -> dict:
    from stable_baselines3 import PPO
    agent = PPO.load(model_path, env=None)

    env = TrafficControlEnv(
        net_file=str(SINGLE_NET),
        route_file=str(SINGLE_ROUTE),
        sim_duration=EVAL_DURATION + 500,
        single_agent=True,
        reward_fn="queue",
        obs_class=IntersectionStateExtractor,
        show_warnings=False,
    )
    obs, _ = env.reset()
    queues, waits, speeds = [], [], []
    action_counts = {}

    for _ in range((EVAL_DURATION // 5) * 4 + 200):
        sumo = env.sumo
        if sumo is None or sumo.simulation.getTime() >= EVAL_DURATION:
            break
        action, _ = agent.predict(obs, deterministic=True)
        act = int(action)
        action_counts[act] = action_counts.get(act, 0) + 1
        obs, _, _, done, _ = env.step(act)
        q, w, s = _collect_step(sumo, env.signal_ids)
        queues.append(q); waits.append(w); speeds.append(s)
        if done: break

    try: env.close()
    except Exception: pass

    total = sum(action_counts.values()) or 1
    phase_str = " ".join(f"P{k}:{v/total*100:.0f}%" for k, v in sorted(action_counts.items()))
    return {
        "mean_queue": float(np.mean(queues)) if queues else 999,
        "mean_wait":  float(np.mean(waits))  if waits  else 999,
        "mean_speed": float(np.mean(speeds)) if speeds else 0,
        "phases":     phase_str,
    }


def find_best_checkpoint(exp_dir: Path) -> tuple[str, int, float]:
    """Trả về (path, step, queue) của checkpoint tốt nhất từ learning curve CSV."""
    curve_csv = exp_dir / "checkpoint_learning_curve.csv"

    best_path  = None
    best_step  = -1
    best_queue = float("inf")

    if curve_csv.exists():
        with open(curve_csv) as f:
            for row in csv.DictReader(f):
                try:
                    step  = int(row["step"])
                    queue = float(row["mean_queue"])
                except (ValueError, KeyError):
                    continue
                if queue < best_queue:
                    best_queue = queue
                    best_step  = step
    else:
        # Fallback: scan checkpoints trực tiếp (không có CSV)
        ckpt_dir = exp_dir / "checkpoints"
        if ckpt_dir.exists():
            for f in ckpt_dir.glob("ppo_ckpt_*_steps.zip"):
                try:
                    step = int(f.stem.split("_steps")[0].split("_")[-1])
                except ValueError:
                    continue
                # không biết queue, lấy step cuối cùng
                if step > best_step:
                    best_step = step
                    best_path = str(f)

    if best_step < 0:
        return None, -1, float("inf")

    # Tìm file checkpoint tương ứng
    if best_step == 999999:   # marker cho "final"
        best_path = str(exp_dir / "ppo_final_model.zip")
    else:
        ckpt = exp_dir / "checkpoints" / f"ppo_ckpt_{best_step}_steps.zip"
        best_path = str(ckpt) if ckpt.exists() else None

    return best_path, best_step, best_queue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root",  default="./experiments")
    ap.add_argument("--save_dir",  default="./results_best_ckpt")
    ap.add_argument("--only",      type=str, default=None)
    ap.add_argument("--eval_duration", type=int, default=7200)
    args = ap.parse_args()

    global EVAL_DURATION
    EVAL_DURATION = args.eval_duration

    root     = Path(args.exp_root)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    only = [x.strip() for x in args.only.split(",")] if args.only else None
    exps = sorted([d for d in root.iterdir()
                   if d.is_dir()
                   and d.name.startswith("ppo_")
                   and not any(d.name.endswith(s) for s in ("_failed","_bak","_old"))
                   and (only is None or any(o in d.name for o in only))])

    print(f"\nTim thay {len(exps)} PPO experiments")
    print(f"Eval duration: {EVAL_DURATION}s | Route: {SINGLE_ROUTE.name}")
    print(f"\n{'Experiment':42s}  {'BestStep':>9}  {'CurveQ':>7}  {'EvalQ':>7}  "
          f"{'Wait':>10}  {'vRandom':>8}  Phases")
    print("-" * 120)

    RANDOM_Q = 63.79
    rows = []
    t0 = time.time()

    for exp in exps:
        ckpt_path, best_step, curve_q = find_best_checkpoint(exp)
        if ckpt_path is None or not Path(ckpt_path).exists():
            print(f"  {exp.name:42s}  [SKIP] khong tim thay checkpoint")
            continue

        step_str = f"{best_step:,}" if best_step < 999999 else "final"
        try:
            m = eval_model(ckpt_path)
        except Exception as e:
            print(f"  {exp.name:42s}  {step_str:>9}  [ERR] {e}")
            continue

        pct = (m["mean_queue"] - RANDOM_Q) / RANDOM_Q * 100
        sign = "✓" if m["mean_queue"] < RANDOM_Q else "✗"
        print(f"  {exp.name:42s}  {step_str:>9}  {curve_q:7.1f}  "
              f"{m['mean_queue']:7.2f}  {m['mean_wait']:10.1f}  "
              f"{pct:+7.1f}% {sign}  {m['phases']}")

        parts = exp.name.split("_")
        seed  = parts[-1]
        mid   = [p for p in parts[1:-1] if p not in ("raw","single")]
        reward = "_".join(mid) if mid else "?"
        rows.append({
            "algo": "ppo", "reward": reward, "seed": seed,
            "best_step": best_step if best_step < 999999 else "final",
            "curve_queue": round(curve_q, 2),
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items() if k != "phases"},
            "phases": m["phases"],
        })

    # Lưu CSV
    if rows:
        out = save_dir / "ppo_best_checkpoint_results.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n→ Luu: {out}")

    print(f"\nHoan tat trong {(time.time()-t0)/60:.1f} phut")


if __name__ == "__main__":
    main()
