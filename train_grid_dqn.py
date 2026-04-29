"""
train_grid_dqn.py — IPPO-style DQN/DDQN tren 2x2 grid (4 junction).

Kien truc: 1 shared DQN/DDQN model (7D obs, Discrete action).
4 agents dung chung weights va replay buffer (IPPO voi weight sharing).

Su dung:
    python train_grid_dqn.py --algo dqn --reward_type queue --seed 42 --save_dir ./exp_grid/dqn_queue_s42
    python train_grid_dqn.py --algo ddqn --reward_type pressure --seed 123 --save_dir ./exp_grid/ddqn_pressure_s123
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback


class StepLogger(BaseCallback):
    def __init__(self, log_freq: int = 10_000):
        super().__init__()
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            print(f"  step {self.num_timesteps}", flush=True)
        return True

if "SUMO_HOME" not in os.environ:
    sys.exit("[train_grid_dqn] SUMO_HOME chua duoc khai bao")

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


class DoubleDQN(DQN):
    """SB3 DQN voi double Q-learning (override target computation)."""
    def train(self, gradient_steps, batch_size=100):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size,
                                                    env=self._vec_normalize_env)
            with __import__("torch").no_grad():
                # Double DQN: chon action bang online network, danh gia bang target
                next_q_online = self.q_net(replay_data.next_observations)
                next_actions  = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = self.q_net_target(replay_data.next_observations)
                next_q_values = next_q_target.gather(1, next_actions)
                target_q = (replay_data.rewards
                            + (1 - replay_data.dones.float())
                            * self.gamma * next_q_values)
            current_q = self.q_net(replay_data.observations)
            current_q = current_q.gather(1, replay_data.actions.long())
            loss = __import__("torch.nn.functional", fromlist=["mse_loss"]).mse_loss(
                current_q, target_q)
            self.policy.optimizer.zero_grad()
            loss.backward()
            __import__("torch").nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(loss.item())
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates",    self._n_updates)
        self.logger.record("train/loss",         float(__import__("numpy").mean(losses)))


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
    p.add_argument("--algo",         default="dqn", choices=["dqn","ddqn"])
    p.add_argument("--reward_type",  default="queue",
                   choices=["queue","pressure","wait-clip"])
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--total_steps",  type=int, default=200_000)
    p.add_argument("--sim_duration", type=int, default=100_000)
    p.add_argument("--buffer_size",  type=int, default=500_000)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--save_dir",     default="./exp_grid/dqn_s42")
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
    algo_label = args.algo.upper()
    print(f"[grid-{algo_label}] {n_agents} agents | reward={args.reward_type} | seed={args.seed}")
    print(f"              obs={vec_env.observation_space.shape} | act={vec_env.action_space}")

    ckpt_cb = CheckpointCallback(
        save_freq = max(args.save_freq // n_agents, 1),
        save_path = str(save_dir / "checkpoints"),
        name_prefix=f"grid_{args.algo}_ckpt",
    )

    AlgoClass = DoubleDQN if args.algo == "ddqn" else DQN
    model = AlgoClass(
        "MlpPolicy", vec_env,
        learning_rate        = args.lr,
        buffer_size          = args.buffer_size,
        learning_starts      = 5_000,
        batch_size           = 128,
        tau                  = 1.0,
        gamma                = 0.99,
        train_freq           = (4, "step"),
        gradient_steps       = 1,
        target_update_interval= 1_000,
        exploration_fraction = 0.15,
        exploration_final_eps= 0.05,
        optimize_memory_usage= False,
        verbose              = 1,
        seed                 = args.seed,
        device               = "cpu",
    )

    callbacks = CallbackList([ckpt_cb, StepLogger(log_freq=10_000)])
    model.learn(total_timesteps=args.total_steps, callback=callbacks,
                progress_bar=False, log_interval=1)
    model.save(str(save_dir / f"grid_{args.algo}_final_model"))
    vec_env.close()
    print(f"[grid-{algo_label}] Done. Saved -> {save_dir}")


if __name__ == "__main__":
    main()
