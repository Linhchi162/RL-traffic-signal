"""
evaluate_all.py — Danh gia toan bo model tren 1 nut giao (single intersection)

Chay moi model trong thu muc experiments/ tren tap test da sinh san,
tong hop ket qua voi mean +- std, xuat CSV san sang cho plotting.

Su dung:
    python evaluate_all.py --models_dir ./experiments --save_dir ./results

Output:
    results/all_results.csv      -- Tat ca ket qua (moi dong = 1 model)
    results/summary.csv          -- Tong hop mean +- std theo (algorithm, reward)
    results/timeseries_results.csv
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Fix encoding Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np

if "SUMO_HOME" not in os.environ:
    sys.exit("[evaluate_all] SUMO_HOME chua duoc khai bao")

import traci
import sumolib
import gymnasium as gym

_HERE = Path(__file__).parent
_RLTSCQ_NETS = _HERE / "RLTSCQ" / "RLTSCQ-main" / "sumo_rl" / "nets" / "RLQ"

SINGLE_NET   = _RLTSCQ_NETS / "caliberated_net.xml"
# Generated test flows (seed=999, never seen during training).
# Falls back to paper test_flows.xml if generated file is missing.
_GENERATED_TEST = _HERE / "generated_flows" / "test_s999.xml"
_route_env   = os.environ.get("EVAL_ROUTE_OVERRIDE")
SINGLE_ROUTE = (
    Path(_route_env) if _route_env
    else (_GENERATED_TEST if _GENERATED_TEST.exists() else _RLTSCQ_NETS / "test_flows.xml")
)

EVAL_DURATION = int(os.environ.get("EVAL_DURATION_OVERRIDE", 7_200))  # giay (~2h sim)

from rl_controller.traffic_env import TrafficControlEnv
from rl_controller.state_builder import IntersectionStateExtractor
from rl_controller.webster import DynamicWebsterController

USE_LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TravelTracker:
    """Track throughput and mean travel time across one evaluation episode."""
    def __init__(self):
        self._dep: dict = {}
        self._tt: list  = []
        self._arrived   = 0
        self._prev_vehs: set = set()

    def update(self, sumo_conn):
        t = sumo_conn.simulation.getTime()
        try:
            for vid in sumo_conn.simulation.getDepartedVehiclesIDs():
                self._dep[vid] = t
            for vid in sumo_conn.simulation.getArrivedVehiclesIDs():
                self._arrived += 1
                if vid in self._dep:
                    self._tt.append(t - self._dep.pop(vid))
        except AttributeError:
            current = set(sumo_conn.vehicle.getIDList())
            for vid in (current - self._prev_vehs):
                self._dep[vid] = t
            for vid in (self._prev_vehs - current):
                self._arrived += 1
                if vid in self._dep:
                    self._tt.append(t - self._dep.pop(vid))
            self._prev_vehs = current

    @property
    def throughput(self):
        return self._arrived

    @property
    def mean_travel_time(self):
        return float(np.mean(self._tt)) if self._tt else 0.0


def collect_step_metrics(sumo_conn, signal_ids: list) -> dict:
    all_vehs = sumo_conn.vehicle.getIDList()
    speeds   = [sumo_conn.vehicle.getSpeed(v) for v in all_vehs] if all_vehs else []
    waits    = [sumo_conn.vehicle.getWaitingTime(v) for v in all_vehs] if all_vehs else []

    total_q    = 0
    lane_queues = {}  # lane_id -> halting count
    for sid in signal_ids:
        lanes = list(dict.fromkeys(sumo_conn.trafficlight.getControlledLanes(sid)))
        for lane in lanes:
            try:
                q = sumo_conn.lane.getLastStepHaltingNumber(lane)
                total_q += q
                lane_queues[lane] = lane_queues.get(lane, 0) + q
            except Exception:
                pass

    n_vehs = max(1, len(all_vehs))
    return {
        "total_queue":       total_q,
        "lane_queues":       lane_queues,
        "vehicles_running":  len(all_vehs),
        "vehicles_stopped":  sum(1 for s in speeds if s < 0.1),
        "total_wait":        sum(waits),
        "mean_wait_per_veh": sum(waits) / n_vehs,
        "mean_speed":        float(np.mean(speeds)) if speeds else 0.0,
    }


def _print_action_distribution(action_counts: dict, label: str):
    total = sum(action_counts.values()) or 1
    print(f"\n  [{label}] Phan phoi action (tong {total} buoc):")
    for act in sorted(action_counts):
        pct = action_counts[act] / total * 100
        bar = "█" * int(pct / 2)
        print(f"    Pha {act}: {pct:5.1f}%  {bar}")


def _print_lane_wait(lane_wait_accum: dict, label: str, top_n: int = 6):
    if not lane_wait_accum:
        return
    sorted_lanes = sorted(lane_wait_accum.items(), key=lambda x: -x[1])
    print(f"\n  [{label}] Top {top_n} lanes co hang cho cao nhat (trung binh xe dung/buoc):")
    for lane, total in sorted_lanes[:top_n]:
        print(f"    {lane}: {total:.1f} xe dung trung binh")


def _empty_result():
    return {"mean_queue": 0, "mean_wait": 0, "mean_wait_per_veh": 0,
            "mean_speed": 0, "throughput": 0, "mean_travel_time": 0, "total_reward": 0}


# ---------------------------------------------------------------------------
# Eval functions
# ---------------------------------------------------------------------------

def run_ppo_eval(model_path: str, obs_mode: str = "raw",
                 _step_out: list = None) -> dict:
    from stable_baselines3 import PPO
    from rl_controller.state_builder import IntersectionStateExtractor, BaselineObservation
    from rl_controller.encoder_obs import (
        CompressedState_4D, CompressedState_8D, CompressedState_16D,
        CompressedState_19D, CompressedState_32D,
    )

    obs_cls = {
        "raw":            IntersectionStateExtractor,
        "baseline":       BaselineObservation,
        "compressed_4d":  CompressedState_4D,
        "compressed_8d":  CompressedState_8D,
        "compressed_16d": CompressedState_16D,
        "compressed_19d": CompressedState_19D,
        "compressed_32d": CompressedState_32D,
        "compressed":     CompressedState_16D,
    }.get(obs_mode, IntersectionStateExtractor)

    agent   = PPO.load(model_path, env=None)
    tracker = _TravelTracker()

    env = TrafficControlEnv(
        net_file=str(SINGLE_NET),
        route_file=str(SINGLE_ROUTE),
        sim_duration=EVAL_DURATION + 500,
        single_agent=True,
        reward_fn="queue",
        obs_class=obs_cls,
        show_warnings=False,
    )

    obs, _ = env.reset()
    metrics_list  = []
    total_reward  = 0.0
    _step_idx     = 0
    action_counts = {}
    lane_wait_acc = {}

    for _ in range(EVAL_DURATION // 5 * 4 + 200):
        sumo_conn = env.sumo
        if sumo_conn is None:
            break
        t = sumo_conn.simulation.getTime()
        if t >= EVAL_DURATION:
            break
        tracker.update(sumo_conn)
        m = collect_step_metrics(sumo_conn, env.signal_ids)
        action, _ = agent.predict(obs, deterministic=True)
        act_int = int(action)
        action_counts[act_int] = action_counts.get(act_int, 0) + 1
        obs, reward, _, done, _ = env.step(act_int)
        total_reward += float(reward)
        metrics_list.append(m)
        for lane, q in m["lane_queues"].items():
            lane_wait_acc[lane] = lane_wait_acc.get(lane, 0.0) + q
        if _step_out is not None:
            _step_out.append({"step": _step_idx, "t": int(t),
                              "action":      act_int,
                              "total_queue": m["total_queue"],
                              "mean_speed":  m["mean_speed"]})
        _step_idx += 1
        if done:
            break

    try:
        env.close()
    except Exception:
        pass

    if not metrics_list:
        return _empty_result()

    # Normalize lane acc to per-step average
    lane_wait_avg = {l: v / max(1, _step_idx) for l, v in lane_wait_acc.items()}
    label = Path(model_path).parent.name
    _print_action_distribution(action_counts, label)
    _print_lane_wait(lane_wait_avg, label)

    return {
        "mean_queue":        float(np.mean([m["total_queue"] for m in metrics_list])),
        "mean_wait":         float(np.mean([m["total_wait"] for m in metrics_list])),
        "mean_wait_per_veh": float(np.mean([m["mean_wait_per_veh"] for m in metrics_list])),
        "mean_speed":        float(np.mean([m["mean_speed"] for m in metrics_list])),
        "throughput":        tracker.throughput,
        "mean_travel_time":  tracker.mean_travel_time,
        "total_reward":      total_reward,
    }


def run_dqn_eval(model_path: str, _step_out: list = None) -> dict:
    from stable_baselines3 import DQN
    from train_dqn import RescoDQNEnv, resco_reward

    agent   = DQN.load(model_path, env=None)
    tracker = _TravelTracker()

    env = RescoDQNEnv(
        net_file=str(SINGLE_NET),
        route_file=str(SINGLE_ROUTE),
        sim_duration=EVAL_DURATION + 500,
    )
    obs, _ = env.reset()

    total_reward  = 0.0
    queue_list, wait_list, speed_list = [], [], []
    _step_idx     = 0
    action_counts = {}
    lane_wait_acc = {}

    for _ in range(EVAL_DURATION // 5 * 4 + 200):
        action, _ = agent.predict(obs, deterministic=True)
        act_int = int(action)
        action_counts[act_int] = action_counts.get(act_int, 0) + 1
        obs, reward, _, done, _ = env.step(act_int)
        total_reward += float(reward)
        try:
            if env.sumo:
                tracker.update(env.sumo)
                m = collect_step_metrics(env.sumo, [env._ts_id])
                queue_list.append(m["total_queue"])
                wait_list.append(m["total_wait"])
                speed_list.append(m["mean_speed"])
                for lane, q in m["lane_queues"].items():
                    lane_wait_acc[lane] = lane_wait_acc.get(lane, 0.0) + q
                if _step_out is not None:
                    _step_out.append({"step": _step_idx,
                                      "action":      act_int,
                                      "total_queue": m["total_queue"],
                                      "mean_speed":  m["mean_speed"]})
        except Exception:
            pass
        _step_idx += 1
        if done:
            break

    try:
        env.close()
    except Exception:
        pass

    lane_wait_avg = {l: v / max(1, _step_idx) for l, v in lane_wait_acc.items()}
    label = Path(model_path).parent.name
    _print_action_distribution(action_counts, label)
    _print_lane_wait(lane_wait_avg, label)

    return {
        "mean_queue":        float(np.mean(queue_list)) if queue_list else 0,
        "mean_wait":         float(np.mean(wait_list))  if wait_list else 0,
        "mean_wait_per_veh": float(np.mean([w / max(1, q) for w, q in zip(wait_list, queue_list)])) if wait_list else 0,
        "mean_speed":        float(np.mean(speed_list)) if speed_list else 0,
        "throughput":        tracker.throughput,
        "mean_travel_time":  tracker.mean_travel_time,
        "total_reward":      total_reward,
    }


def run_webster_eval(_step_out: list = None) -> dict:
    label = f"wsb_{int(time.time()*1000) % 100000}"
    cmd = [
        sumolib.checkBinary("sumo"),
        "-n", str(SINGLE_NET),
        "-r", str(SINGLE_ROUTE),
        "--time-to-teleport", "-1",
        "--no-warnings",
    ]
    if USE_LIBSUMO:
        traci.start(cmd)
        sumo = traci
    else:
        traci.start(cmd, label=label)
        sumo = traci.getConnection(label)

    signal_ids = list(sumo.trafficlight.getIDList())
    tracker = _TravelTracker()

    def get_green_phases(ts):
        logics = sumo.trafficlight.getAllProgramLogics(ts)
        if not logics:
            return [0]
        return [i for i, ph in enumerate(logics[0].phases)
                if "G" in ph.state.upper() and "Y" not in ph.state.upper()]

    controllers = {
        ts: DynamicWebsterController(ts, get_green_phases(ts), sumo)
        for ts in signal_ids
    }

    metrics_list = []
    _step_idx = 0
    try:
        while sumo.simulation.getTime() < EVAL_DURATION:
            sumo.simulationStep()
            t = sumo.simulation.getTime()
            tracker.update(sumo)
            for ctrl in controllers.values():
                ctrl.step(t)
            m = collect_step_metrics(sumo, signal_ids)
            metrics_list.append(m)
            if _step_out is not None:
                _step_out.append({"step": _step_idx,
                                  "total_queue": m["total_queue"],
                                  "mean_speed":  m["mean_speed"]})
            _step_idx += 1
    except Exception as exc:
        print(f"  [Webster] Loi: {exc}")
    finally:
        try:
            if not USE_LIBSUMO:
                traci.switch(label)
            traci.close()
        except Exception:
            pass

    if not metrics_list:
        return _empty_result()

    return {
        "mean_queue":        float(np.mean([m["total_queue"] for m in metrics_list])),
        "mean_wait":         float(np.mean([m["total_wait"] for m in metrics_list])),
        "mean_wait_per_veh": float(np.mean([m["mean_wait_per_veh"] for m in metrics_list])),
        "mean_speed":        float(np.mean([m["mean_speed"] for m in metrics_list])),
        "throughput":        tracker.throughput,
        "mean_travel_time":  tracker.mean_travel_time,
        "total_reward":      float(np.mean([-m["total_queue"] for m in metrics_list])),
    }


def run_random_eval(n_actions: int = 4, seed: int = 0, _step_out: list = None) -> dict:
    """Random policy baseline — chon pha ngau nhien de xem PPO co beat duoc khong."""
    import random
    rng = random.Random(seed)

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
    tracker = _TravelTracker()
    metrics_list  = []
    action_counts = {}
    _step_idx = 0

    for _ in range(EVAL_DURATION // 5 * 4 + 200):
        sumo_conn = env.sumo
        if sumo_conn is None:
            break
        if sumo_conn.simulation.getTime() >= EVAL_DURATION:
            break
        tracker.update(sumo_conn)
        m = collect_step_metrics(sumo_conn, env.signal_ids)
        act = rng.randint(0, n_actions - 1)
        action_counts[act] = action_counts.get(act, 0) + 1
        obs, _, _, done, _ = env.step(act)
        metrics_list.append(m)
        if _step_out is not None:
            _step_out.append({"step": _step_idx, "t": int(sumo_conn.simulation.getTime()),
                              "action": act, "total_queue": m["total_queue"],
                              "mean_speed": m["mean_speed"]})
        _step_idx += 1
        if done:
            break

    try:
        env.close()
    except Exception:
        pass

    if not metrics_list:
        return _empty_result()

    _print_action_distribution(action_counts, f"random_seed{seed}")
    return {
        "mean_queue":        float(np.mean([m["total_queue"] for m in metrics_list])),
        "mean_wait":         float(np.mean([m["total_wait"] for m in metrics_list])),
        "mean_wait_per_veh": float(np.mean([m["mean_wait_per_veh"] for m in metrics_list])),
        "mean_speed":        float(np.mean([m["mean_speed"] for m in metrics_list])),
        "throughput":        tracker.throughput,
        "mean_travel_time":  tracker.mean_travel_time,
        "total_reward":      float(np.mean([-m["total_queue"] for m in metrics_list])),
    }


def run_fixed_eval(_step_out: list = None) -> dict:
    env = TrafficControlEnv(
        net_file=str(SINGLE_NET),
        route_file=str(SINGLE_ROUTE),
        sim_duration=EVAL_DURATION + 500,
        single_agent=True,
        reward_fn="queue",
        obs_class=IntersectionStateExtractor,
        fixed_signal=True,
        show_warnings=False,
    )

    obs, _ = env.reset()
    tracker = _TravelTracker()
    metrics_list = []
    _step_idx = 0

    for _ in range(EVAL_DURATION // 5 * 4 + 200):
        sumo_conn = env.sumo
        if sumo_conn is None:
            break
        if sumo_conn.simulation.getTime() >= EVAL_DURATION:
            break
        tracker.update(sumo_conn)
        m = collect_step_metrics(sumo_conn, env.signal_ids)
        obs, _, _, done, _ = env.step(0)
        metrics_list.append(m)
        if _step_out is not None:
            _step_out.append({"step": _step_idx,
                              "total_queue": m["total_queue"],
                              "mean_speed":  m["mean_speed"]})
        _step_idx += 1
        if done:
            break

    try:
        env.close()
    except Exception:
        pass

    if not metrics_list:
        return _empty_result()

    return {
        "mean_queue":        float(np.mean([m["total_queue"] for m in metrics_list])),
        "mean_wait":         float(np.mean([m["total_wait"] for m in metrics_list])),
        "mean_wait_per_veh": float(np.mean([m["mean_wait_per_veh"] for m in metrics_list])),
        "mean_speed":        float(np.mean([m["mean_speed"] for m in metrics_list])),
        "throughput":        tracker.throughput,
        "mean_travel_time":  tracker.mean_travel_time,
        "total_reward":      float(np.mean([-m["total_queue"] for m in metrics_list])),
    }


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def _infer_obs_mode(name: str) -> str:
    n = name.lower()
    for d in (4, 8, 16, 19, 32):
        if f"ae-{d}d" in n or f"ae_{d}d" in n or f"compressed_{d}d" in n:
            return f"compressed_{d}d"
    if "baseline" in n:
        return "baseline"
    return "raw"


_AE_KEYWORDS = ("ae-", "ae_", "compressed_")


def discover_models(models_dir: Path, skip_ae: bool = False) -> List[Tuple[str, str, str, str, str]]:
    """
    Quet thu muc experiments/ de tim tat ca model da train.

    Returns:
        List of (algo, reward, seed, model_path, obs_mode)
    """
    results = []
    if not models_dir.exists():
        return results

    for subdir in sorted(models_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name

        if any(name.endswith(sfx) for sfx in ("_failed", "_bak", "_old", "_backup")):
            continue
        if skip_ae and any(k in name.lower() for k in _AE_KEYWORDS):
            continue

        ppo_path = subdir / "ppo_final_model.zip"
        dqn_path = subdir / "dqn_final_model.zip"

        if ppo_path.exists():
            parts = name.split("_")
            if len(parts) >= 3:
                algo     = parts[0]
                seed     = parts[-1]
                obs_mode = _infer_obs_mode(name)
                middle   = [p for p in parts[1:-1]
                            if p not in ("raw", "baseline", "single")]
                reward   = "_".join(middle) if middle else "queue"
                results.append((algo, reward, seed, str(ppo_path), obs_mode))

        elif dqn_path.exists():
            parts  = name.split("_")
            algo   = parts[0]
            seed   = parts[-1]
            middle = [p for p in parts[1:-1] if p not in ("single", "resco")]
            reward = "-".join(middle) if middle else "wait-clip"
            results.append((algo, reward, seed, str(dqn_path), "resco"))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Danh gia toan bo model — 1 nut giao",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models_dir", default="./experiments")
    p.add_argument("--save_dir",   default="./results")
    p.add_argument("--route", type=str, default=None,
                   help="Duong dan toi file flow test (override SINGLE_ROUTE)")
    p.add_argument("--skip_fixed",   action="store_true")
    p.add_argument("--skip_webster", action="store_true")
    p.add_argument("--skip_random",  action="store_true")
    p.add_argument("--skip_ae",      action="store_true")
    p.add_argument("--only", type=str, default=None,
                   help="Chi eval cac model co ten khop (phan cach bang dau phay). "
                        "Ket qua se duoc merge vao CSV hien co.")
    return p.parse_args()


def main():
    args = parse_args()
    models_dir = Path(args.models_dir)
    save_dir   = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Cho phep override route qua --route hoac env var
    global SINGLE_ROUTE
    if args.route:
        SINGLE_ROUTE = Path(args.route)

    if not SINGLE_NET.exists():
        print(f"[WARN] Thieu file mang: {SINGLE_NET}")
    if not SINGLE_ROUTE.exists():
        print(f"[WARN] Thieu file flow: {SINGLE_ROUTE}")

    models = discover_models(models_dir, skip_ae=args.skip_ae)

    only_filter = None
    if args.only:
        only_filter = [x.strip() for x in args.only.split(",") if x.strip()]
        models = [m for m in models if Path(m[3]).parts[-2] in only_filter]
        print(f"\n[--only] Loc con {len(models)} model(s): {only_filter}")

    print(f"\nTim thay {len(models)} model(s) trong {models_dir}")
    print(f"Test flow : {SINGLE_ROUTE.name}")
    print(f"Eval time : {EVAL_DURATION}s\n")

    all_csv = save_dir / "all_results.csv"
    fields = ["algo", "reward", "obs_mode", "seed",
              "mean_queue", "mean_wait", "mean_wait_per_veh", "mean_speed",
              "throughput", "mean_travel_time", "total_reward"]
    all_rows = []
    ts_rows  = []

    total_jobs = len(models)
    if not args.skip_fixed:
        total_jobs += 1
    if not args.skip_webster:
        total_jobs += 1
    if not args.skip_random:
        total_jobs += 1
    job_idx = 0
    t0 = time.time()

    # Random policy baseline
    if not args.skip_random:
        job_idx += 1
        print(f"[{job_idx}/{total_jobs}] Random policy baseline (4 pha, seed=0)")
        try:
            _sl = []
            m   = run_random_eval(n_actions=4, seed=0, _step_out=_sl)
            all_rows.append({"algo": "random", "reward": "-", "obs_mode": "-", "seed": "-", **m})
            for s in _sl:
                ts_rows.append({"algo": "random", "reward": "-", "obs_mode": "-", "seed": "-", **s})
            print(f"  queue={m['mean_queue']:.2f}  wait={m['mean_wait']:.1f}  speed={m['mean_speed']:.3f}")
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"  [ERR] {exc}")

    # Fixed-time baseline
    if not args.skip_fixed:
        job_idx += 1
        print(f"[{job_idx}/{total_jobs}] Fixed-time baseline")
        try:
            _sl = []
            m   = run_fixed_eval(_step_out=_sl)
            all_rows.append({"algo": "fixed", "reward": "-", "obs_mode": "-", "seed": "-", **m})
            for s in _sl:
                ts_rows.append({"algo": "fixed", "reward": "-", "obs_mode": "-", "seed": "-", **s})
            print(f"  queue={m['mean_queue']:.2f}  wait={m['mean_wait']:.1f}  speed={m['mean_speed']:.3f}")
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"  [ERR] {exc}")

    # Webster baseline
    if not args.skip_webster:
        job_idx += 1
        print(f"[{job_idx}/{total_jobs}] Webster baseline")
        try:
            _sl = []
            m   = run_webster_eval(_step_out=_sl)
            all_rows.append({"algo": "webster", "reward": "-", "obs_mode": "-", "seed": "-", **m})
            for s in _sl:
                ts_rows.append({"algo": "webster", "reward": "-", "obs_mode": "-", "seed": "-", **s})
            print(f"  queue={m['mean_queue']:.2f}  wait={m['mean_wait']:.1f}  speed={m['mean_speed']:.3f}")
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"  [ERR] {exc}")

    # Trained models
    for algo, reward, seed, model_path, obs_mode in models:
        job_idx += 1
        print(f"[{job_idx}/{total_jobs}] {algo.upper()} | reward={reward} | obs={obs_mode} | seed={seed}")
        try:
            _sl = []
            if algo in ("dqn", "ddqn"):
                m = run_dqn_eval(model_path, _step_out=_sl)
            else:
                m = run_ppo_eval(model_path, obs_mode=obs_mode, _step_out=_sl)
            all_rows.append({"algo": algo, "reward": reward, "obs_mode": obs_mode, "seed": seed, **m})
            for s in _sl:
                ts_rows.append({"algo": algo, "reward": reward, "obs_mode": obs_mode, "seed": seed, **s})
            print(f"  queue={m['mean_queue']:.2f}  wait={m['mean_wait']:.1f}  speed={m['mean_speed']:.3f}")
        except Exception as exc:
            import traceback
            print(f"  [ERR] {exc}")
            traceback.print_exc()

    # Merge với CSV cũ nếu dùng --only
    if only_filter and all_csv.exists():
        existing = []
        with open(all_csv, newline="") as f:
            for row in csv.DictReader(f):
                model_dir = row.get("algo","") + "_" + row.get("reward","") + "_" + row.get("seed","")
                if not any(f in model_dir for f in only_filter):
                    existing.append(row)
        all_rows = existing + all_rows

    with open(all_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    # Summary
    summary_csv = save_dir / "summary.csv"
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in all_rows:
        key = (row["algo"], row["reward"], row.get("obs_mode", "-"))
        grouped[key].append(row)

    summary_rows = []
    for (algo, reward, obs_mode), rows in sorted(grouped.items()):
        queues = [float(r["mean_queue"]) for r in rows]
        waits  = [float(r["mean_wait"])  for r in rows]
        wpv    = [float(r.get("mean_wait_per_veh", 0)) for r in rows]
        speeds = [float(r["mean_speed"]) for r in rows]
        thru   = [float(r.get("throughput", 0)) for r in rows]
        ttime  = [float(r.get("mean_travel_time", 0)) for r in rows]
        summary_rows.append({
            "algo": algo, "reward": reward, "obs_mode": obs_mode,
            "n_runs": len(rows),
            "mean_queue":        round(np.mean(queues),  3),
            "std_queue":         round(np.std(queues),   3),
            "median_queue":      round(np.median(queues),3),
            "mean_wait":         round(np.mean(waits),   2),
            "std_wait":          round(np.std(waits),    2),
            "mean_wait_per_veh": round(np.mean(wpv),     2),
            "mean_speed":        round(np.mean(speeds),  4),
            "std_speed":         round(np.std(speeds),   4),
            "mean_throughput":   round(np.mean(thru),    1),
            "std_throughput":    round(np.std(thru),     1),
            "mean_travel_time":  round(np.mean(ttime),   2),
            "std_travel_time":   round(np.std(ttime),    2),
        })

    sum_fields = ["algo", "reward", "obs_mode", "n_runs",
                  "mean_queue", "std_queue", "median_queue",
                  "mean_wait", "std_wait", "mean_wait_per_veh",
                  "mean_speed", "std_speed",
                  "mean_throughput", "std_throughput",
                  "mean_travel_time", "std_travel_time"]
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        w.writerows(summary_rows)

    if ts_rows:
        ts_csv    = save_dir / "timeseries_results.csv"
        ts_fields = ["algo", "reward", "obs_mode", "seed", "step", "action", "total_queue", "mean_speed"]
        with open(ts_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ts_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(ts_rows)
        print(f"  Timeseries: {ts_csv}")

    elapsed = round((time.time() - t0) / 60, 1)
    print(f"\n{'='*55}")
    print(f"  HOAN TAT DANH GIA ({elapsed} phut)")
    print(f"  Chi tiet : {all_csv}")
    print(f"  Tom tat  : {summary_csv}")
    print(f"{'='*55}")
    print(f"\nBuoc tiep theo:")
    print(f"  python plot_results.py --results_dir {save_dir}")


if __name__ == "__main__":
    main()
