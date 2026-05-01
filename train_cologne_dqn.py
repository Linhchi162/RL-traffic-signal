"""
train_cologne_dqn.py — IPPO-style DQN/DDQN tren mang Cologne that.

Kien truc giong het train_grid_dqn.py (7D obs, Discrete(4), shared weights).
Chi khac net_file / route_file.

Su dung:
    python train_cologne_dqn.py --net_file nets/cologne3/cologne3.net.xml \
        --route_file nets/cologne3/cologne3_train.rou.xml \
        --algo ddqn --reward_type wait-clip --seed 42 \
        --save_dir ./exp_cologne3/ddqn_wait-clip_s42
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

if "SUMO_HOME" not in os.environ:
    sys.exit("[train_cologne_dqn] SUMO_HOME chua duoc khai bao")

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from rl_controller.traffic_env   import TrafficControlEnv
from rl_controller.state_builder import BaselineObservation
from rl_controller.grid_env      import MultiAgentVecEnv


class StepLogger(BaseCallback):
    def __init__(self, log_freq: int = 5_000, total_steps: int = 200_000,
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
            eps      = getattr(self.model, "exploration_rate", float("nan"))
            upd      = getattr(self.model, "_n_updates", 0)
            pct      = 100 * t / self.total_steps
            mean_rew = float(np.mean(self._rew_buf)) if self._rew_buf else float("nan")
            print(f"  step {t:>7d}/{self.total_steps} ({pct:5.1f}%)"
                  f"  eps={eps:.3f}  updates={upd:>6d}"
                  f"  mean_rew={mean_rew:+.3f}", flush=True)
            self._next_log = t + self.log_freq
        return True


class DoubleDQN(DQN):
    """SB3 DQN voi double Q-learning."""
    def train(self, gradient_steps, batch_size=100):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size,
                                                    env=self._vec_normalize_env)
            with __import__("torch").no_grad():
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
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/loss",      float(__import__("numpy").mean(losses)))


def make_env(net_file, route_file, reward_type, seed, sim_duration, sumo_begin=0, gui=False):
    # sim_duration = relative duration (s); TrafficControlEnv.sim_max_time = absolute end time
    sim_end = sumo_begin + sim_duration
    extra   = f"--begin {sumo_begin}" if sumo_begin > 0 else None
    def _make():
        return TrafficControlEnv(
            net_file         = net_file,
            route_file       = route_file,
            sim_duration     = sim_end,
            reward_fn        = reward_type,
            obs_class        = BaselineObservation,
            sumo_seed        = seed,
            single_agent     = False,
            use_gui          = gui,
            show_warnings    = False,
            extra_sumo_args  = extra,
        )
    return _make


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--net_file",      required=True)
    p.add_argument("--route_file",    required=True)
    p.add_argument("--algo",          default="ddqn", choices=["dqn", "ddqn"])
    p.add_argument("--reward_type",   default="wait-clip",
                   choices=["queue", "pressure", "wait-clip"])
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--total_steps",   type=int,   default=200_000)
    p.add_argument("--sim_duration",  type=int,   default=100_000,
                   help="Do dai episode (s). Voi cologne real: 3600.")
    p.add_argument("--sumo_begin",    type=int,   default=0,
                   help="Thoi diem bat dau tuyet doi trong route file (s). Voi cologne real: 25200.")
    p.add_argument("--buffer_size",   type=int,   default=500_000)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--save_dir",      default="./exp_cologne3/ddqn_s42")
    p.add_argument("--save_freq",     type=int,   default=20_000)
    p.add_argument("--gui",           action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vec_env  = MultiAgentVecEnv(make_env(
        args.net_file, args.route_file,
        args.reward_type, args.seed, args.sim_duration, args.sumo_begin, args.gui,
    ))
    n_agents    = vec_env.num_envs
    algo_label  = args.algo.upper()
    net_name    = Path(args.net_file).name.split(".")[0]
    print(f"[{net_name}-{algo_label}] {n_agents} agents | reward={args.reward_type} | seed={args.seed}")
    print(f"  obs={vec_env.observation_space.shape} | act={vec_env.action_space}")

    ckpt_cb = CheckpointCallback(
        save_freq  = max(args.save_freq // n_agents, 1),
        save_path  = str(save_dir / "checkpoints"),
        name_prefix= f"{net_name}_{args.algo}_ckpt",
    )

    # Scale buffer theo so agent: moi step VecEnv luu n_agents transitions.
    # buffer_size=500k / 3 agents → buffer day tai ~167k step → gay dip.
    # Nhan voi n_agents de dam bao buffer tuong duong 500k step thuc te.
    scaled_buffer = args.buffer_size * n_agents
    print(f"  buffer_size={args.buffer_size} × {n_agents} agents = {scaled_buffer}")

    AlgoClass = DoubleDQN if args.algo == "ddqn" else DQN
    model = AlgoClass(
        "MlpPolicy", vec_env,
        learning_rate         = args.lr,
        buffer_size           = scaled_buffer,
        learning_starts       = 5_000,
        batch_size            = 128,
        tau                   = 1.0,
        gamma                 = 0.99,
        train_freq            = (4, "step"),
        gradient_steps        = 1,
        target_update_interval= 1_000,
        exploration_fraction  = 0.15,
        exploration_final_eps = 0.05,
        optimize_memory_usage = False,
        verbose               = 1,
        seed                  = args.seed,
        device                = "cpu",
    )

    callbacks = CallbackList([ckpt_cb, StepLogger(log_freq=5_000, total_steps=args.total_steps)])
    model.learn(total_timesteps=args.total_steps, callback=callbacks,
                progress_bar=False, log_interval=1)
    model.save(str(save_dir / f"{net_name}_{args.algo}_final_model"))
    vec_env.close()
    print(f"[{net_name}-{algo_label}] Done. Saved -> {save_dir}")


if __name__ == "__main__":
    main()
