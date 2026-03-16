"""
evaluate_ppo.py — Danh gia agent PPO da huan luyen

Su dung:
    # Danh gia 1 nut giao (single)
    python evaluate_ppo.py --model_path ./out_single/ppo_final_model.zip --save_dir ./eval_single

    # Danh gia 4 nut giao (multi)
    python evaluate_ppo.py --model_path ./out_multi/ppo_final_model.zip --save_dir ./eval_multi --mode multi

Tham so:
    --model_path  : Duong dan file model .zip (bat buoc)
    --save_dir    : Thu muc luu ket qua (bat buoc)
    --mode        : single | multi (mac dinh: single)
    --duration    : Thoi gian mo phong (giay, mac dinh: 5000)
    --obs_mode    : raw | compressed | baseline (mac dinh: raw)
    --gui         : Bat SUMO GUI
    --fixed       : Chay baseline co dinh (khong dung model)
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Fix encoding Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
from stable_baselines3 import PPO

if "SUMO_HOME" not in os.environ:
    sys.exit("[evaluate_ppo] SUMO_HOME chua duoc khai bao")

_HERE = Path(__file__).parent
_RLTSCQ_NETS = _HERE / "RLTSCQ" / "RLTSCQ-main" / "sumo_rl" / "nets" / "RLQ"

SINGLE_NET   = _RLTSCQ_NETS / "caliberated_net.xml"
SINGLE_ROUTE = _RLTSCQ_NETS / "test_flows.xml"
MULTI_NET    = _HERE / "sumo_nets" / "grid2x2.net.xml"
MULTI_ROUTE  = _HERE / "sumo_nets" / "grid2x2_test.rou.xml"

from rl_controller.traffic_env import TrafficControlEnv
from rl_controller.state_builder import IntersectionStateExtractor, BaselineObservation
from rl_controller.encoder_obs import CompressedState_16D


# ---------------------------------------------------------------------------
# Wrapper multi-agent -> gym.Env (tai su dung tu train_ppo.py)
# ---------------------------------------------------------------------------

import gymnasium as gym

class MultiAgentGymWrapper(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env: TrafficControlEnv):
        super().__init__()
        self._env = env
        self.observation_space = env.observation_space
        self.action_space      = env.action_space
        self._obs_queue: list  = []
        self._reward_accum: dict = {}
        self._done = False

    @property
    def unwrapped(self):
        return self._env

    def reset(self, seed=None, **kwargs):
        obs_dict, info = self._env.reset()
        self._obs_queue = list(obs_dict.items())
        self._done = False
        self._reward_accum = {sid: 0.0 for sid in self._env.signal_ids}
        if self._obs_queue:
            _sid, obs = self._obs_queue.pop(0)
            return obs, info
        return np.zeros(self.observation_space.shape, dtype=np.float32), info

    def step(self, action):
        if self._obs_queue:
            _sid, next_obs = self._obs_queue.pop(0)
            reward = float(np.mean(list(self._reward_accum.values()))) if self._reward_accum else 0.0
            return next_obs, reward, False, self._done, {}

        actions = {
            sid: action
            for sid in self._env.signal_ids
            if self._env.controllers[sid].time_to_act
        }
        obs_dict, reward_dict, done_dict, info = self._env.step(actions)
        self._reward_accum = {k: float(v) for k, v in reward_dict.items()}
        self._done = done_dict.get("__all__", False)
        if obs_dict:
            self._obs_queue = list(obs_dict.items())
            _sid, obs = self._obs_queue.pop(0)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        mean_reward = float(np.mean(list(self._reward_accum.values()))) if self._reward_accum else 0.0
        return obs, mean_reward, False, self._done, {}

    def close(self):
        self._env.close()

    def render(self):
        return self._env.render()


# ---------------------------------------------------------------------------
# Cac ham tinh metric
# ---------------------------------------------------------------------------

def collect_metrics(sumo_conn, signal_ids: list) -> dict:
    """
    Thu thap metric tu toan he thong:
    - Tong xe dang chay
    - Tong xe dung (hang cho)
    - Tong thoi gian cho
    - Toc do trung binh
    """
    all_vehs = sumo_conn.vehicle.getIDList()
    if not all_vehs:
        return {
            "vehicles_running": 0,
            "vehicles_stopped": 0,
            "total_wait": 0.0,
            "mean_speed": 0.0,
            "total_queue": 0,
        }

    speeds = [sumo_conn.vehicle.getSpeed(v) for v in all_vehs]
    waits  = [sumo_conn.vehicle.getWaitingTime(v) for v in all_vehs]

    # Tong hang cho tai tat ca nut giao
    total_q = 0
    for sid in signal_ids:
        lanes = list(dict.fromkeys(sumo_conn.trafficlight.getControlledLanes(sid)))
        for lane in lanes:
            try:
                total_q += sumo_conn.lane.getLastStepHaltingNumber(lane)
            except Exception:
                pass

    return {
        "vehicles_running": len(all_vehs),
        "vehicles_stopped": sum(1 for s in speeds if s < 0.1),
        "total_wait": sum(waits),
        "mean_speed": float(np.mean(speeds)),
        "total_queue": total_q,
    }


# ---------------------------------------------------------------------------
# Evaluate loop
# ---------------------------------------------------------------------------

def run_evaluation(
    eval_env,
    agent,
    is_multi: bool,
    duration: int,
    csv_path: str,
    is_fixed: bool = False,
):
    """
    Chay vong lap danh gia, ghi metric vao CSV moi buoc.

    Args:
        eval_env: Moi truong (don hoac boc multi)
        agent: PPO agent (None neu is_fixed=True)
        is_multi: True = 4 nut giao
        duration: Thoi gian mo phong (giay)
        csv_path: Duong dan ghi CSV
        is_fixed: True = khong dung RL, de den chay co dinh
    """
    base_env = eval_env.unwrapped if hasattr(eval_env, "unwrapped") else eval_env
    signal_ids = base_env.signal_ids

    obs, _ = eval_env.reset()

    fields = [
        "sim_time", "vehicles_running", "vehicles_stopped",
        "total_queue", "total_wait", "mean_speed",
        "step_reward",
    ]
    f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    total_reward = 0.0
    n_steps = 0
    step_rewards = []

    try:
        while True:
            sumo_conn = base_env.sumo
            if sumo_conn is None:
                break

            sim_time = sumo_conn.simulation.getTime()
            if sim_time >= duration:
                break

            # Lay metric truoc khi buoc
            m = collect_metrics(sumo_conn, signal_ids)

            if is_fixed:
                obs, reward, terminated, truncated, _ = eval_env.step(0)
            else:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)

            total_reward += float(reward)
            step_rewards.append(float(reward))
            n_steps += 1

            row = {
                "sim_time": round(sim_time, 1),
                **m,
                "step_reward": round(float(reward), 4),
            }
            writer.writerow(row)

            if terminated or truncated:
                break

    except KeyboardInterrupt:
        print("\n[evaluate] Dung boi nguoi dung.")
    except Exception as exc:
        import traceback
        traceback.print_exc()
    finally:
        f.close()

    return {
        "n_steps": n_steps,
        "total_reward": total_reward,
        "mean_reward": total_reward / max(n_steps, 1),
        "mean_queue": np.mean([r for r in step_rewards]) if step_rewards else 0,
    }


# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Danh gia PPO dieu khien den tin hieu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path", required=True, help="Duong dan file model .zip")
    p.add_argument("--save_dir",   required=True, help="Thu muc luu ket qua")
    p.add_argument("--mode", default="single", choices=["single", "multi"],
                   help="single=1 nut giao | multi=4 nut giao")
    p.add_argument("--obs_mode", default="raw", choices=["raw", "compressed", "baseline"],
                   help="Loai observation (phai trung voi khi train)")
    p.add_argument("--duration", type=int, default=5000, help="Thoi gian mo phong (giay)")
    p.add_argument("--gui", action="store_true", default=False)
    p.add_argument("--min_green", type=int, default=10)
    p.add_argument("--max_green", type=int, default=40)
    p.add_argument("--amber_sec", type=int, default=5)
    p.add_argument("--step_interval", type=int, default=5)
    p.add_argument("--fixed", action="store_true", default=False,
                   help="Chay fixed-time baseline (khong dung model)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    is_multi = (args.mode == "multi")
    os.makedirs(args.save_dir, exist_ok=True)

    # Chon file SUMO
    if is_multi:
        net_file   = str(MULTI_NET)
        route_file = str(MULTI_ROUTE)
    else:
        net_file   = str(SINGLE_NET)
        route_file = str(SINGLE_ROUTE)

    # Chon obs class
    if args.obs_mode == "compressed":
        obs_cls = CompressedState_16D
    elif args.obs_mode == "baseline":
        obs_cls = BaselineObservation
    else:
        obs_cls = IntersectionStateExtractor

    # Tai model
    agent = None
    if not args.fixed:
        print(f"Nap model: {args.model_path}")
        agent = PPO.load(args.model_path, env=None)

    # CSV output
    label = "fixed" if args.fixed else "rl"
    csv_path = os.path.join(args.save_dir, f"eval_{label}_{args.mode}.csv")

    print("=" * 55)
    print(f"  Danh gia PPO - {'Fixed-time' if args.fixed else 'RL agent'}")
    print(f"  Mode     : {args.mode} ({'4 nut giao' if is_multi else '1 nut giao'})")
    print(f"  Obs mode : {args.obs_mode}")
    print(f"  Duration : {args.duration}s")
    print(f"  Net file : {Path(net_file).name}")
    print(f"  Output   : {csv_path}")
    print("=" * 55)

    # Khoi tao moi truong
    base_env = TrafficControlEnv(
        net_file=net_file,
        route_file=route_file,
        use_gui=args.gui,
        sim_duration=args.duration + 500,
        min_green=args.min_green,
        max_green=args.max_green,
        amber_sec=args.amber_sec,
        step_interval=args.step_interval,
        single_agent=not is_multi,
        reward_fn="queue",
        obs_class=obs_cls,
        fixed_signal=args.fixed,
    )

    eval_env = MultiAgentGymWrapper(base_env) if is_multi else base_env

    t0 = time.time()
    result = run_evaluation(
        eval_env=eval_env,
        agent=agent,
        is_multi=is_multi,
        duration=args.duration,
        csv_path=csv_path,
        is_fixed=args.fixed,
    )
    elapsed = time.time() - t0

    try:
        eval_env.close()
    except Exception:
        pass

    print(f"\n{'='*55}")
    print(f"  KET QUA DANH GIA ({elapsed:.1f}s)")
    print(f"{'='*55}")
    print(f"  Steps thu thap : {result['n_steps']:,}")
    print(f"  Reward trung binh per step : {result['mean_reward']:.4f}")
    print(f"  Total reward   : {result['total_reward']:.2f}")
    print(f"  CSV da luu     : {csv_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
