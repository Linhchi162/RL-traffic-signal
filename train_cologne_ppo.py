"""
train_cologne_ppo.py — IPPO-style PPO tren mang Cologne that.

Kien truc giong het train_grid_ppo.py (7D obs, Discrete action, shared weights).
Chi khac net_file / route_file.

Su dung:
    python train_cologne_ppo.py --net_file nets/cologne3/cologne3.net.xml \
        --route_file nets/cologne3/cologne3.rou.xml \
        --reward_type wait-clip --seed 42 \
        --save_dir ./exp_cologne3_real/ppo_wait-clip_s42
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

if "SUMO_HOME" not in os.environ:
    sys.exit("[train_cologne_ppo] SUMO_HOME chua duoc khai bao")

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from rl_controller.traffic_env   import TrafficControlEnv
from rl_controller.state_builder import BaselineObservation
from rl_controller.grid_env      import MultiAgentVecEnv


class StepLogger(BaseCallback):
    def __init__(self, log_freq: int = 5_000, total_steps: int = 500_000,
                 window: int = 200):
        super().__init__()
        self.log_freq    = log_freq
        self.total_steps = total_steps
        self.window      = window
        self._next_log   = log_freq
        self._rew_buf    = []

    def _on_step(self) -> bool:
        rews = self.locals.get("rewards")
        if rews is not None:
            self._rew_buf.extend(float(r) for r in rews)
            if len(self._rew_buf) > self.window:
                self._rew_buf = self._rew_buf[-self.window:]
        t = self.num_timesteps
        if t >= self._next_log:
            pct      = 100 * t / self.total_steps
            mean_rew = float(np.mean(self._rew_buf)) if self._rew_buf else float("nan")
            print(f"  step {t:>7d}/{self.total_steps} ({pct:5.1f}%)"
                  f"  mean_rew={mean_rew:+.3f}", flush=True)
            self._next_log = t + self.log_freq
        return True


def make_env(net_file, route_file, reward_type, seed, sim_duration, sumo_begin=0, gui=False):
    sim_end = sumo_begin + sim_duration
    extra   = f"--begin {sumo_begin}" if sumo_begin > 0 else None
    def _make():
        return TrafficControlEnv(
            net_file        = net_file,
            route_file      = route_file,
            sim_duration    = sim_end,
            reward_fn       = reward_type,
            obs_class       = BaselineObservation,
            sumo_seed       = seed,
            single_agent    = False,
            use_gui         = gui,
            show_warnings   = False,
            extra_sumo_args = extra,
        )
    return _make


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--net_file",      required=True)
    p.add_argument("--route_file",    required=True)
    p.add_argument("--reward_type",   default="wait-clip",
                   choices=["queue", "pressure", "wait-clip"])
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--total_steps",   type=int,   default=500_000)
    p.add_argument("--sim_duration",  type=int,   default=3_600,
                   help="Do dai episode (s). Voi cologne real: 3600.")
    p.add_argument("--sumo_begin",    type=int,   default=0,
                   help="Thoi diem bat dau tuyet doi trong route file (s). Voi cologne real: 25200.")
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--save_dir",      default="./exp_cologne3_real/ppo_s42")
    p.add_argument("--save_freq",     type=int,   default=50_000)
    p.add_argument("--gui",           action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vec_env = MultiAgentVecEnv(make_env(
        args.net_file, args.route_file,
        args.reward_type, args.seed, args.sim_duration, args.sumo_begin, args.gui,
    ))
    n_agents = vec_env.num_envs
    net_name = Path(args.net_file).name.split(".")[0]
    print(f"[{net_name}-PPO] {n_agents} agents | reward={args.reward_type} | seed={args.seed}")
    print(f"  obs={vec_env.observation_space.shape} | act={vec_env.action_space}")

    ckpt_cb = CheckpointCallback(
        save_freq   = max(args.save_freq // n_agents, 1),
        save_path   = str(save_dir / "checkpoints"),
        name_prefix = f"{net_name}_ppo_ckpt",
    )

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate = args.lr,
        n_steps       = 512,
        batch_size    = 128,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        verbose       = 1,
        seed          = args.seed,
        device        = "cpu",
    )

    callbacks = CallbackList([ckpt_cb, StepLogger(log_freq=5_000, total_steps=args.total_steps)])
    model.learn(total_timesteps=args.total_steps, callback=callbacks,
                progress_bar=False)
    model.save(str(save_dir / f"{net_name}_ppo_final_model"))
    vec_env.close()
    print(f"[{net_name}-PPO] Done. Saved -> {save_dir}")


if __name__ == "__main__":
    main()
