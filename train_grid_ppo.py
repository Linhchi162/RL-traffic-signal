"""
train_grid_ppo.py — IPPO voi shared policy tren 2x2 grid (4 junction).

Kien truc: 1 PPO model (7D obs, Discrete action), 4 agents dung chung weights.
Moi junction la 1 'env' trong MultiAgentVecEnv.

Su dung:
    python train_grid_ppo.py --reward_type queue --seed 42 --save_dir ./exp_grid/ppo_queue_s42
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback


class StepLogger(BaseCallback):
    def __init__(self, log_freq: int = 5_000, total_steps: int = 200_000):
        super().__init__()
        self.log_freq    = log_freq
        self.total_steps = total_steps

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t % self.log_freq == 0 and t > 0:
            pct  = 100 * t / self.total_steps
            rews = self.locals.get("rewards")
            mean_rew = float(np.mean(rews)) if rews is not None else float("nan")
            print(f"  step {t:>7d}/{self.total_steps} ({pct:5.1f}%)"
                  f"  mean_rew={mean_rew:+.3f}", flush=True)
        return True

if "SUMO_HOME" not in os.environ:
    sys.exit("[train_grid_ppo] SUMO_HOME chua duoc khai bao")

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_HERE = Path(__file__).parent
GRID_NET   = _HERE / "nets" / "grid2x2.net.xml"
GRID_ROUTE = _HERE / "generated_flows" / "grid_s0.xml"

from rl_controller.traffic_env import TrafficControlEnv
from rl_controller.state_builder import BaselineObservation
from rl_controller.grid_env import MultiAgentVecEnv


def make_env(reward_type, seed, sim_duration, gui=False):
    def _make():
        return TrafficControlEnv(
            net_file    = str(GRID_NET),
            route_file  = str(GRID_ROUTE),
            sim_duration= sim_duration,
            reward_fn   = reward_type,
            obs_class   = BaselineObservation,
            sumo_seed   = seed,
            single_agent= False,
            use_gui     = gui,
            show_warnings=False,
        )
    return _make


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reward_type",  default="queue",
                   choices=["queue","pressure","wait-clip"])
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--total_steps",  type=int, default=200_000)
    p.add_argument("--sim_duration", type=int, default=100_000)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--save_dir",     default="./exp_grid/ppo_s42")
    p.add_argument("--save_freq",    type=int, default=20_000)
    p.add_argument("--gui",          action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vec_env = MultiAgentVecEnv(make_env(args.reward_type, args.seed,
                                        args.sim_duration, args.gui))

    n_agents = vec_env.num_envs
    print(f"[grid-PPO] {n_agents} agents | reward={args.reward_type} | seed={args.seed}")
    print(f"           obs={vec_env.observation_space.shape} | "
          f"act={vec_env.action_space}")

    ckpt_cb = CheckpointCallback(
        save_freq = max(args.save_freq // n_agents, 1),
        save_path = str(save_dir / "checkpoints"),
        name_prefix="grid_ppo_ckpt",
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
    model.save(str(save_dir / "grid_ppo_final_model"))
    vec_env.close()
    print(f"[grid-PPO] Done. Saved -> {save_dir}")


if __name__ == "__main__":
    main()
