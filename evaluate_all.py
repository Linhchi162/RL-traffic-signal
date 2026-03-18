"""
evaluate_all.py — Danh gia toan bo model tren 3 muc luu luong

Chay moi model trong thu muc experiments/ tren 3 kich ban luu luong,
tong hop ket qua voi mean +- std, xuat CSV san sang cho plotting.

Su dung:
    python evaluate_all.py --models_dir ./experiments --save_dir ./results

Output:
    results/all_results.csv      -- Tat ca ket qua (moi dong = 1 model x 1 flow)
    results/summary.csv          -- Tong hop mean +- std theo (algorithm, flow)
    results/cycle_data/          -- Du lieu Queue vs Cycle cho Fig 5a
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# --- Paper RLTSCQ setup (single intersection) ---
SINGLE_NET   = _RLTSCQ_NETS / "caliberated_net.xml"
SINGLE_ROUTE = _RLTSCQ_NETS / "test_flows.xml"   # paper test_flows

# --- Multi-intersection (grid 2x2) ---
MULTI_NET  = _HERE / "sumo_nets" / "grid2x2.net.xml"
MULTI_FLOW_FILES = {
    "high":   _HERE / "sumo_nets" / "grid2x2_test_high.rou.xml",
    "medium": _HERE / "sumo_nets" / "grid2x2_test_medium.rou.xml",
    "low":    _HERE / "sumo_nets" / "grid2x2_test_low.rou.xml",
}
EVAL_DURATION = int(os.environ.get("EVAL_DURATION_OVERRIDE", 5_000))  # giay

from rl_controller.traffic_env import TrafficControlEnv
from rl_controller.state_builder import IntersectionStateExtractor
from rl_controller.webster import DynamicWebsterController

USE_LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


# ---------------------------------------------------------------------------
# Wrapper multi-agent (tai su dung)
# ---------------------------------------------------------------------------

class _MultiWrapper(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env):
        super().__init__()
        self._env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._obs_queue = []
        self._reward_accum = {}
        self._done = False

    @property
    def unwrapped(self):
        return self._env

    def reset(self, seed=None, **kwargs):
        obs_dict, info = self._env.reset()
        self._obs_queue = list(obs_dict.items())
        self._done = False
        self._reward_accum = {s: 0.0 for s in self._env.signal_ids}
        if self._obs_queue:
            _, obs = self._obs_queue.pop(0)
            return obs, info
        return np.zeros(self.observation_space.shape, dtype=np.float32), info

    def step(self, action):
        if self._obs_queue:
            _, next_obs = self._obs_queue.pop(0)
            r = float(np.mean(list(self._reward_accum.values()))) if self._reward_accum else 0.0
            return next_obs, r, False, self._done, {}
        actions = {
            sid: action for sid in self._env.signal_ids
            if self._env.controllers[sid].time_to_act
        }
        obs_dict, reward_dict, done_dict, info = self._env.step(actions)
        self._reward_accum = {k: float(v) for k, v in reward_dict.items()}
        self._done = done_dict.get("__all__", False)
        if obs_dict:
            self._obs_queue = list(obs_dict.items())
            _, obs = self._obs_queue.pop(0)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        mean_r = float(np.mean(list(self._reward_accum.values()))) if self._reward_accum else 0.0
        return obs, mean_r, False, self._done, {}

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Cac ham danh gia
# ---------------------------------------------------------------------------

class _TravelTracker:
    """Track throughput and mean travel time across one evaluation episode."""
    def __init__(self):
        self._dep: dict = {}
        self._tt: list = []
        self._arrived = 0
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
            # Fallback: so sanh tap vehicle IDs giua cac buoc
            current = set(sumo_conn.vehicle.getIDList())
            for vid in (current - self._prev_vehs):   # xe moi xuat hien
                self._dep[vid] = t
            for vid in (self._prev_vehs - current):   # xe bien mat = da den
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
    """Thu thap metric tai 1 buoc mo phong."""
    all_vehs = sumo_conn.vehicle.getIDList()
    speeds = [sumo_conn.vehicle.getSpeed(v) for v in all_vehs] if all_vehs else []
    waits  = [sumo_conn.vehicle.getWaitingTime(v) for v in all_vehs] if all_vehs else []

    total_q = 0
    for sid in signal_ids:
        lanes = list(dict.fromkeys(sumo_conn.trafficlight.getControlledLanes(sid)))
        for lane in lanes:
            try:
                total_q += sumo_conn.lane.getLastStepHaltingNumber(lane)
            except Exception:
                pass

    n_vehs = max(1, len(all_vehs))
    return {
        "total_queue": total_q,
        "vehicles_running": len(all_vehs),
        "vehicles_stopped": sum(1 for s in speeds if s < 0.1),
        "total_wait": sum(waits),
        "mean_wait_per_veh": sum(waits) / n_vehs,
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
    }


def run_ppo_eval(model_path: str, flow_key: str, obs_mode: str = "raw",
                 scope: str = "single", _step_out: list = None) -> dict:
    """
    Chay 1 episode voi PPO model, tra ve metrics tong hop.

    Args:
        scope: "single" (caliberated_net + test_flows.xml — paper RLTSCQ)
               "multi"  (grid2x2 + flow_key=high/medium/low)

    Returns:
        dict voi mean_queue, mean_wait, mean_speed, total_reward
    """
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

    is_single = (scope == "single")
    if is_single:
        net_file = str(SINGLE_NET)
        route_file = str(SINGLE_ROUTE)
    else:
        net_file = str(MULTI_NET)
        route_file = str(MULTI_FLOW_FILES[flow_key])

    agent = PPO.load(model_path, env=None)
    tracker = _TravelTracker()

    base_env = TrafficControlEnv(
        net_file=net_file,
        route_file=route_file,
        sim_duration=EVAL_DURATION + 500,
        single_agent=is_single,
        reward_fn="queue",
        obs_class=obs_cls,
        show_warnings=False,
    )

    if is_single:
        obs, _ = base_env.reset()
    else:
        env = _MultiWrapper(base_env)
        obs, _ = env.reset()

    metrics_list = []
    total_reward = 0.0
    _step_idx = 0

    for _ in range(EVAL_DURATION // 5 * 4 + 200):
        sumo_conn = base_env.sumo
        if sumo_conn is None:
            break
        t = sumo_conn.simulation.getTime()
        if t >= EVAL_DURATION:
            break
        tracker.update(sumo_conn)
        m = collect_step_metrics(sumo_conn, base_env.signal_ids)
        action, _ = agent.predict(obs, deterministic=True)
        if is_single:
            obs, reward, _, done, _ = base_env.step(int(action))
        else:
            obs, reward, _, done, _ = env.step(action)
        total_reward += float(reward)
        metrics_list.append(m)
        if _step_out is not None:
            _step_out.append({"step": _step_idx, "t": int(t),
                              "total_queue": m["total_queue"],
                              "mean_speed":  m["mean_speed"]})
        _step_idx += 1
        if done:
            break

    try:
        if is_single:
            base_env.close()
        else:
            env.close()
    except Exception:
        pass

    if not metrics_list:
        return {"mean_queue": 0, "mean_wait": 0, "mean_speed": 0, "total_reward": 0}

    n_signals = max(1, len(base_env.signal_ids))
    return {
        "mean_queue":       float(np.mean([m["total_queue"] for m in metrics_list])) / n_signals,
        "mean_wait":        float(np.mean([m["total_wait"] for m in metrics_list])),
        "mean_wait_per_veh":float(np.mean([m["mean_wait_per_veh"] for m in metrics_list])),
        "mean_speed":       float(np.mean([m["mean_speed"] for m in metrics_list])),
        "throughput":       tracker.throughput,
        "mean_travel_time": tracker.mean_travel_time,
        "total_reward":     total_reward,
    }


def run_dqn_eval(model_path: str, flow_key: str, scope: str = "single",
                 _step_out: list = None) -> dict:
    """Chay 1 episode voi DQN model.

    Args:
        scope: "single" (caliberated_net) | "multi" (grid2x2 4 nut)
    """
    from stable_baselines3 import DQN
    from train_dqn import RescoDQNEnv, MultiDQNWrapper, resco_reward

    is_single = (scope == "single")
    if is_single:
        net_file = str(SINGLE_NET)
        route_file = str(SINGLE_ROUTE)
    else:
        net_file = str(MULTI_NET)
        route_file = str(MULTI_FLOW_FILES[flow_key])

    agent = DQN.load(model_path, env=None)
    tracker = _TravelTracker()

    if is_single:
        single_env = RescoDQNEnv(
            net_file=net_file,
            route_file=route_file,
            ts_index=0,
            sim_duration=EVAL_DURATION + 500,
        )
        envs = [single_env]
        env = single_env
    else:
        envs = [
            RescoDQNEnv(
                net_file=net_file,
                route_file=route_file,
                ts_index=i,
                sim_duration=EVAL_DURATION + 500,
            )
            for i in range(4)
        ]
        env = MultiDQNWrapper(envs)
    obs, _ = env.reset()
    total_reward = 0.0
    queue_list = []
    wait_list = []
    speed_list = []

    _step_idx = 0
    for _ in range(EVAL_DURATION // 5 * 4 + 200):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, _, done, _ = env.step(action)
        total_reward += float(reward)
        # Lay metric tu tat ca envs (moi env quan ly 1 nut giao)
        try:
            q_vals, w_vals, s_vals = [], [], []
            if envs[0].sumo:
                tracker.update(envs[0].sumo)
            for e in envs:
                if e.sumo:
                    m = collect_step_metrics(e.sumo, [e._ts_id])
                    q_vals.append(m["total_queue"])
                    w_vals.append(m["total_wait"])
                    s_vals.append(m["mean_speed"])
            if q_vals:
                mq = float(np.mean(q_vals))
                ms = float(np.mean(s_vals))
                queue_list.append(mq)
                wait_list.append(float(np.mean(w_vals)))
                speed_list.append(ms)
                if _step_out is not None:
                    _step_out.append({"step": _step_idx,
                                      "total_queue": mq, "mean_speed": ms})
        except Exception:
            pass
        _step_idx += 1
        if done:
            break

    try:
        env.close()
    except Exception:
        pass

    return {
        "mean_queue":       float(np.mean(queue_list)) if queue_list else 0,
        "mean_wait":        float(np.mean(wait_list)) if wait_list else 0,
        "mean_wait_per_veh":float(np.mean([w / max(1, q) for w, q in zip(wait_list, queue_list)])) if wait_list else 0,
        "mean_speed":       float(np.mean(speed_list)) if speed_list else 0,
        "throughput":       tracker.throughput,
        "mean_travel_time": tracker.mean_travel_time,
        "total_reward":     total_reward,
    }


def run_webster_eval(flow_key: str, scope: str = "single",
                     _step_out: list = None) -> dict:
    """Chay Webster dynamic baseline."""
    if scope == "single":
        net_file = str(SINGLE_NET)
        route_file = str(SINGLE_ROUTE)
    else:
        net_file = str(MULTI_NET)
        route_file = str(MULTI_FLOW_FILES[flow_key])

    label = f"wsb_{int(time.time()*1000) % 100000}"
    cmd = [
        sumolib.checkBinary("sumo"),
        "-n", net_file,
        "-r", route_file,
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

    # Phat hien pha xanh cho tung nut
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
        return {"mean_queue": 0, "mean_wait": 0, "mean_speed": 0, "total_reward": 0}

    n_signals = max(1, len(signal_ids))
    return {
        "mean_queue":       float(np.mean([m["total_queue"] for m in metrics_list])) / n_signals,
        "mean_wait":        float(np.mean([m["total_wait"] for m in metrics_list])),
        "mean_wait_per_veh":float(np.mean([m["mean_wait_per_veh"] for m in metrics_list])),
        "mean_speed":       float(np.mean([m["mean_speed"] for m in metrics_list])),
        "throughput":       tracker.throughput,
        "mean_travel_time": tracker.mean_travel_time,
        "total_reward":     float(np.mean([-m["total_queue"] for m in metrics_list])),
    }


def run_fixed_eval(flow_key: str, scope: str = "single",
                   _step_out: list = None) -> dict:
    """Chay fixed-time baseline (SUMO default timing)."""
    is_single = (scope == "single")
    if is_single:
        net_file = str(SINGLE_NET)
        route_file = str(SINGLE_ROUTE)
    else:
        net_file = str(MULTI_NET)
        route_file = str(MULTI_FLOW_FILES[flow_key])

    base_env = TrafficControlEnv(
        net_file=net_file,
        route_file=route_file,
        sim_duration=EVAL_DURATION + 500,
        single_agent=is_single,
        reward_fn="queue",
        obs_class=IntersectionStateExtractor,
        fixed_signal=True,
        show_warnings=False,
    )

    if is_single:
        obs, _ = base_env.reset()
    else:
        env = _MultiWrapper(base_env)
        obs, _ = env.reset()

    tracker = _TravelTracker()
    metrics_list = []
    _step_idx = 0

    for _ in range(EVAL_DURATION // 5 * 4 + 200):
        sumo_conn = base_env.sumo
        if sumo_conn is None:
            break
        if sumo_conn.simulation.getTime() >= EVAL_DURATION:
            break
        tracker.update(sumo_conn)
        m = collect_step_metrics(sumo_conn, base_env.signal_ids)
        if is_single:
            obs, _, _, done, _ = base_env.step(0)
        else:
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
        if is_single:
            base_env.close()
        else:
            env.close()
    except Exception:
        pass

    if not metrics_list:
        return {"mean_queue": 0, "mean_wait": 0, "mean_speed": 0, "total_reward": 0}

    n_signals = max(1, len(base_env.signal_ids))
    return {
        "mean_queue":       float(np.mean([m["total_queue"] for m in metrics_list])) / n_signals,
        "mean_wait":        float(np.mean([m["total_wait"] for m in metrics_list])),
        "mean_wait_per_veh":float(np.mean([m["mean_wait_per_veh"] for m in metrics_list])),
        "mean_speed":       float(np.mean([m["mean_speed"] for m in metrics_list])),
        "throughput":       tracker.throughput,
        "mean_travel_time": tracker.mean_travel_time,
        "total_reward":     float(np.mean([-m["total_queue"] for m in metrics_list])),
    }


# ---------------------------------------------------------------------------
# Discovery models
# ---------------------------------------------------------------------------

def _infer_obs_mode(name: str) -> str:
    """Suy lu\u1eadn obs_mode t\u1eeb t\u00ean th\u01b0 m\u1ee5c model."""
    n = name.lower()
    # ppo_queue-AE-16d_s42 / ppo_queue-AE-4d_s42 ...
    for d in (4, 8, 16, 19, 32):
        if f"ae-{d}d" in n or f"ae_{d}d" in n or f"compressed_{d}d" in n:
            return f"compressed_{d}d"
    if "baseline" in n:
        return "baseline"
    return "raw"


_AE_KEYWORDS = ("ae-", "ae_", "compressed_")

def discover_models(models_dir: Path, skip_ae: bool = False) -> List[Tuple[str, str, str, str, str, str]]:
    """
    Quet thu muc experiments/ de tim tat ca model da train.

    Returns:
        List of (algo, reward, seed, model_path, obs_mode, train_scope)
        train_scope: "single" neu ten thu muc chua "single", nguoc lai "multi"
    """
    results = []
    if not models_dir.exists():
        return results

    for subdir in sorted(models_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name  # e.g. "ppo_queue_s42", "dqn_s123"

        # Bo qua thu muc backup (ket thuc bang _failed, _bak, _old)
        if any(name.endswith(sfx) for sfx in ("_failed", "_bak", "_old", "_backup")):
            continue

        if skip_ae and any(k in name.lower() for k in _AE_KEYWORDS):
            continue

        ppo_path = subdir / "ppo_final_model.zip"
        dqn_path = subdir / "dqn_final_model.zip"

        train_scope = "single" if "single" in name.lower() else "multi"

        if ppo_path.exists():
            parts = name.split("_")
            if len(parts) >= 3:
                algo = parts[0]
                seed = parts[-1]          # s42
                obs_mode = _infer_obs_mode(name)

                # Loc bo cac suffix obs khoi phan reward: "raw", "baseline", "single", "multi"
                middle = parts[1:-1]
                middle = [p for p in middle if p not in ("raw", "baseline", "single", "multi")]
                reward = "_".join(middle) if middle else "queue"

                results.append((algo, reward, seed, str(ppo_path), obs_mode, train_scope))
        elif dqn_path.exists():
            parts = name.split("_")
            algo = parts[0]  # "dqn" hoac "ddqn"
            seed = parts[-1]  # "s42"
            # Suy luan reward tu cac phan giua (bo scope keywords)
            middle = parts[1:-1]
            middle = [p for p in middle if p not in ("single", "multi", "resco")]
            reward = "-".join(middle) if middle else "wait-clip"
            results.append((algo, reward, seed, str(dqn_path), "resco", train_scope))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Danh gia toan bo model tren 3 muc luu luong",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models_dir", default="./experiments",
                   help="Thu muc chua ket qua train")
    p.add_argument("--save_dir", default="./results",
                   help="Thu muc luu ket qua danh gia")
    p.add_argument("--skip_fixed", action="store_true",
                   help="Bo qua fixed-time baseline")
    p.add_argument("--skip_webster", action="store_true",
                   help="Bo qua Webster baseline")
    p.add_argument("--flows", nargs="+", default=["high", "medium", "low"],
                   choices=["high", "medium", "low"],
                   help="(Chi cho scope=multi) Cac muc luu luong can test")
    p.add_argument("--scope", default="multi",
                   choices=["single", "multi", "both"],
                   help="single = paper RLTSCQ (caliberated_net + test_flows.xml); "
                        "multi = grid 2x2; both = chay ca hai")
    p.add_argument("--skip_ae", action="store_true",
                   help="Bo qua cac model autoencoder (AE) khi danh gia")
    p.add_argument("--only", type=str, default=None,
                   help="Chi eval cac model co ten khop (phan cach bang dau phay). "
                        "VD: ppo_queue_s42,ppo_pressure_s777. "
                        "Ket qua se duoc merge vao CSV hien co.")
    return p.parse_args()


def _scope_flow_keys(scope: str, multi_flows: list) -> list:
    """Tra ve danh sach (scope, flow_key) can chay."""
    if scope == "single":
        # Single intersection — chi 1 flow file (test_flows.xml)
        return [("single", "paper")]
    if scope == "multi":
        return [("multi", f) for f in multi_flows]
    # both
    return [("single", "paper")] + [("multi", f) for f in multi_flows]


def main():
    args = parse_args()
    models_dir = Path(args.models_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Kiem tra file SUMO theo scope
    if args.scope in ("single", "both"):
        if not SINGLE_NET.exists() or not SINGLE_ROUTE.exists():
            print(f"[WARN] Thieu file paper: {SINGLE_NET} / {SINGLE_ROUTE}")
    if args.scope in ("multi", "both"):
        for flow_key in args.flows:
            rf = MULTI_FLOW_FILES[flow_key]
            if not rf.exists():
                print(f"[WARN] Chua co {rf.name} — chay: python sumo_nets/generate_test_flows.py")

    # Tim tat ca models
    models = discover_models(models_dir, skip_ae=args.skip_ae)

    # Loc neu co --only
    only_filter = None
    if args.only:
        only_filter = [x.strip() for x in args.only.split(",") if x.strip()]
        models = [m for m in models
                  if Path(m[3]).parts[-2] in only_filter]
        print(f"\n[--only] Loc con {len(models)} model(s): {only_filter}")

    print(f"\nTim thay {len(models)} model(s) trong {models_dir}")
    print(f"Scope eval : {args.scope}\n")

    # Danh sach (scope, flow_key) can chay
    scope_flows = _scope_flow_keys(args.scope, args.flows)

    # CSV output
    all_csv = save_dir / "all_results.csv"
    fields = ["algo", "reward", "obs_mode", "seed", "scope", "flow",
              "mean_queue", "mean_wait", "mean_wait_per_veh", "mean_speed",
              "throughput", "mean_travel_time", "total_reward"]
    all_rows = []
    ts_rows  = []   # timeseries: 1 row per simulation step

    n_sf = len(scope_flows)
    total_jobs = len(models) * n_sf
    if not args.skip_fixed:
        total_jobs += n_sf
    if not args.skip_webster:
        total_jobs += n_sf

    job_idx = 0
    t0 = time.time()

    # -------------------------------------------------------
    # Baseline: Fixed-time
    # -------------------------------------------------------
    if not args.skip_fixed:
        for sc, flow_key in scope_flows:
            job_idx += 1
            print(f"[{job_idx}/{total_jobs}] Fixed-time | scope={sc} | flow={flow_key}")
            try:
                _sl = []
                m = run_fixed_eval(flow_key, scope=sc, _step_out=_sl)
                row = {"algo": "fixed", "reward": "-", "obs_mode": "-", "seed": "-",
                       "scope": sc, "flow": flow_key, **m}
                all_rows.append(row)
                for s in _sl:
                    ts_rows.append({"algo": "fixed", "reward": "-", "obs_mode": "-",
                                    "seed": "-", "flow": flow_key, **s})
                print(f"  queue={m['mean_queue']:.2f}  wait={m['mean_wait']:.1f}  speed={m['mean_speed']:.3f}")
            except Exception as exc:
                import traceback; traceback.print_exc()
                print(f"  [ERR] {exc}")

    # -------------------------------------------------------
    # Baseline: Webster
    # -------------------------------------------------------
    if not args.skip_webster:
        for sc, flow_key in scope_flows:
            job_idx += 1
            print(f"[{job_idx}/{total_jobs}] Webster | scope={sc} | flow={flow_key}")
            try:
                _sl = []
                m = run_webster_eval(flow_key, scope=sc, _step_out=_sl)
                row = {"algo": "webster", "reward": "-", "obs_mode": "-", "seed": "-",
                       "scope": sc, "flow": flow_key, **m}
                all_rows.append(row)
                for s in _sl:
                    ts_rows.append({"algo": "webster", "reward": "-", "obs_mode": "-",
                                    "seed": "-", "flow": flow_key, **s})
                print(f"  queue={m['mean_queue']:.2f}  wait={m['mean_wait']:.1f}  speed={m['mean_speed']:.3f}")
            except Exception as exc:
                import traceback; traceback.print_exc()
                print(f"  [ERR] {exc}")

    # -------------------------------------------------------
    # Tat ca trained models
    # -------------------------------------------------------
    for algo, reward, seed, model_path, obs_mode, train_scope in models:
        # Model single chi eval tren single; model multi chi eval tren multi
        model_sf = [(sc, fk) for sc, fk in scope_flows if sc == train_scope]
        if not model_sf:
            model_sf = scope_flows  # fallback neu khong co scope phu hop
        for sc, flow_key in model_sf:
            job_idx += 1
            print(f"[{job_idx}/{total_jobs}] {algo.upper()} | reward={reward} | obs={obs_mode} | seed={seed} | scope={sc} | flow={flow_key}")
            try:
                _sl = []
                if algo in ("dqn", "ddqn"):
                    m = run_dqn_eval(model_path, flow_key, scope=sc, _step_out=_sl)
                else:
                    m = run_ppo_eval(model_path, flow_key, obs_mode=obs_mode, scope=sc, _step_out=_sl)
                row = {"algo": algo, "reward": reward, "obs_mode": obs_mode,
                       "seed": seed, "scope": sc, "flow": flow_key, **m}
                all_rows.append(row)
                for s in _sl:
                    ts_rows.append({"algo": algo, "reward": reward, "obs_mode": obs_mode,
                                    "seed": seed, "flow": flow_key, **s})
                print(f"  queue={m['mean_queue']:.2f}  wait={m['mean_wait']:.1f}  speed={m['mean_speed']:.3f}")
            except Exception as exc:
                import traceback
                print(f"  [ERR] {exc}")
                traceback.print_exc()

    # -------------------------------------------------------
    # Ghi CSV (merge neu dung --only)
    # -------------------------------------------------------
    if only_filter and all_csv.exists():
        # Doc rows cu, bo rows cua cac model vua eval lai, them rows moi
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

    # -------------------------------------------------------
    # Tong hop mean +- std
    # -------------------------------------------------------
    summary_csv = save_dir / "summary.csv"
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in all_rows:
        key = (row["algo"], row["reward"], row.get("obs_mode", "-"), row["scope"], row["flow"])
        grouped[key].append(row)

    summary_rows = []
    for (algo, reward, obs_mode, scope, flow), rows in sorted(grouped.items()):
        queues  = [float(r["mean_queue"]) for r in rows]
        waits   = [float(r["mean_wait"])  for r in rows]
        wpv     = [float(r.get("mean_wait_per_veh", 0)) for r in rows]
        speeds  = [float(r["mean_speed"]) for r in rows]
        thru    = [float(r.get("throughput", 0)) for r in rows]
        ttime   = [float(r.get("mean_travel_time", 0)) for r in rows]
        summary_rows.append({
            "algo": algo, "reward": reward, "obs_mode": obs_mode,
            "scope": scope, "flow": flow,
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

    sum_fields = ["algo","reward","obs_mode","scope","flow","n_runs",
                  "mean_queue","std_queue","median_queue",
                  "mean_wait","std_wait","mean_wait_per_veh",
                  "mean_speed","std_speed",
                  "mean_throughput","std_throughput",
                  "mean_travel_time","std_travel_time"]
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        w.writerows(summary_rows)

    # -------------------------------------------------------
    # Timeseries CSV (de ve line chart)
    # -------------------------------------------------------
    if ts_rows:
        ts_csv = save_dir / "timeseries_results.csv"
        ts_fields = ["algo", "reward", "obs_mode", "seed", "flow",
                     "step", "total_queue", "mean_speed"]
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
    print("\nBuoc tiep theo:")
    print(f"  python plot_multi_figures.py --results_dir {save_dir}")


if __name__ == "__main__":
    main()
