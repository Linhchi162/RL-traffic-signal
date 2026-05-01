"""
train_cologne_curriculum.py — Curriculum learning cho Cologne3 / Cologne8.

Hai che do:
  Mac dinh  : 4 giai doan 25->50->75->100% tren route file that (scaled subsample).
  --rand    : 4 giai doan tren route file SYNTHETIC (random OD, cung luong).
              Phai chay make_random_routes.py truoc de tao _rand_*.rou.xml.
              Eval tren route that -> train/test separation thuc su.

Su dung:
    # Curriculum tren du lieu that (tranh local optima):
    python train_cologne_curriculum.py \\
        --net_file nets/cologne3/cologne3.net.xml \\
        --route_file nets/cologne3/cologne3.rou.xml \\
        --algo ddqn --reward_type wait-clip --seed 42 \\
        --save_dir ./exp_cologne3_curr/ddqn_wait-clip_s42

    # Curriculum tren du lieu synthetic (train/test separation):
    python train_cologne_curriculum.py \\
        --net_file nets/cologne8/cologne8.net.xml \\
        --route_file nets/cologne8/cologne8.rou.xml \\
        --algo ddqn --reward_type wait-clip --seed 42 \\
        --save_dir ./exp_cologne8_rand_curr/ddqn_wait-clip_s42 --rand
"""

import argparse
import os
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

if "SUMO_HOME" not in os.environ:
    sys.exit("[train_cologne_curriculum] SUMO_HOME chua duoc khai bao")

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from rl_controller.traffic_env   import TrafficControlEnv
from rl_controller.state_builder import BaselineObservation
from rl_controller.grid_env      import MultiAgentVecEnv


# ── Curriculum schedule ───────────────────────────────────────────────────────
# (demand_fraction, portion_of_total_steps)
CURRICULUM = [
    (0.25, 0.15),
    (0.50, 0.20),
    (0.75, 0.25),
    (1.00, 0.40),
]


# ── Route helpers ────────────────────────────────────────────────────────────

def rand_route_path(route_file: Path, fraction: float) -> Path:
    """Tra ve duong dan den random route file (phai ton tai truoc)."""
    stem = route_file.stem
    for suffix in (".rou", ".net"):
        stem = stem.replace(suffix, "")
    pct = int(round(fraction * 100))
    return route_file.parent / f"{stem}_rand_{pct}pct.rou.xml"


def ensure_scaled_route(route_file: Path, fraction: float,
                        begin: float, end: float, seed: int) -> Path:
    """Return path to scaled route file, creating it if not present."""
    if fraction >= 1.0:
        return route_file

    stem = route_file.stem
    for suffix in (".rou", ".net"):
        stem = stem.replace(suffix, "")
    pct = int(round(fraction * 100))
    out = route_file.parent / f"{stem}_{pct}pct.rou.xml"

    if out.exists():
        print(f"  [route] {out.name} da co.", flush=True)
        return out

    print(f"  [route] Tao {out.name} ({pct}% demand)...", flush=True)
    tree = ET.parse(str(route_file))
    root = tree.getroot()

    all_vehs  = [c for c in root if c.tag == "vehicle"]
    in_window = [v for v in all_vehs
                 if begin <= float(v.get("depart", 0)) <= end]

    rng    = random.Random(seed)
    n_keep = max(1, round(len(in_window) * fraction))
    keep   = {id(v) for v in rng.sample(in_window, n_keep)}

    for v in in_window:
        if id(v) not in keep:
            root.remove(v)

    tree.write(str(out), encoding="unicode", xml_declaration=True)
    print(f"  [route] {n_keep}/{len(in_window)} vehicles in window -> {out.name}", flush=True)
    return out


# ── DDQN ─────────────────────────────────────────────────────────────────────

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


# ── StepLogger ───────────────────────────────────────────────────────────────

class StepLogger(BaseCallback):
    def __init__(self, log_freq: int = 5_000, total_steps: int = 500_000,
                 window: int = 200, algo: str = "dqn"):
        super().__init__()
        self.log_freq    = log_freq
        self.total_steps = total_steps
        self.window      = window
        self.algo        = algo
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
            if self.algo in ("dqn", "ddqn"):
                eps = getattr(self.model, "exploration_rate", float("nan"))
                upd = getattr(self.model, "_n_updates", 0)
                print(f"  step {t:>7d}/{self.total_steps} ({pct:5.1f}%)"
                      f"  eps={eps:.3f}  updates={upd:>6d}"
                      f"  mean_rew={mean_rew:+.3f}", flush=True)
            else:
                print(f"  step {t:>7d}/{self.total_steps} ({pct:5.1f}%)"
                      f"  mean_rew={mean_rew:+.3f}", flush=True)
            self._next_log = t + self.log_freq
        return True


# ── Env factory ──────────────────────────────────────────────────────────────

def make_env_fn(net_file, route_file, reward_type, seed, sim_duration, sumo_begin, gui=False):
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


# ── Arg parse ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--net_file",     required=True)
    p.add_argument("--route_file",   required=True,
                   help="Full-demand (100%%) route file.")
    p.add_argument("--algo",         default="ddqn", choices=["dqn", "ddqn", "ppo"])
    p.add_argument("--reward_type",  default="wait-clip",
                   choices=["queue", "pressure", "wait-clip"])
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--total_steps",  type=int,   default=500_000)
    p.add_argument("--sim_duration", type=int,   default=3_600,
                   help="Episode duration (s). Cologne real: 3600.")
    p.add_argument("--sumo_begin",   type=int,   default=25_200,
                   help="Absolute start time in route file (s). Cologne real: 25200.")
    p.add_argument("--window_begin", type=float, default=25_200.0,
                   help="Demand window start for route scaling (same as sumo_begin).")
    p.add_argument("--window_end",   type=float, default=28_800.0,
                   help="Demand window end for route scaling.")
    p.add_argument("--buffer_size",  type=int,   default=500_000,
                   help="DQN/DDQN replay buffer size (scaled by n_agents internally).")
    p.add_argument("--lr",           type=float, default=None,
                   help="Learning rate (default: 1e-4 DQN/DDQN, 3e-4 PPO).")
    p.add_argument("--save_dir",     default="./exp_cologne3_curr/ddqn_s42")
    p.add_argument("--save_freq",    type=int,   default=25_000)
    p.add_argument("--clear_buffer", action="store_true",
                   help="Clear DQN/DDQN replay buffer at each stage transition.")
    p.add_argument("--rand",         action="store_true",
                   help="Dung random synthetic routes thay vi scaled real routes. "
                        "Phai chay make_random_routes.py truoc.")
    p.add_argument("--gui",          action="store_true")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    route_file = Path(args.route_file)
    net_file   = str(args.net_file)
    net_name   = Path(net_file).name.split(".")[0]
    algo       = args.algo
    lr         = args.lr or (1e-4 if algo in ("dqn", "ddqn") else 3e-4)

    # ── Compute per-stage step counts ─────────────────────────────────────────
    stage_steps = [(frac, max(1, round(args.total_steps * portion)))
                   for frac, portion in CURRICULUM]
    # Fix rounding drift so sum == total_steps exactly
    diff = args.total_steps - sum(s for _, s in stage_steps)
    if diff:
        last_frac, last_s = stage_steps[-1]
        stage_steps[-1] = (last_frac, last_s + diff)

    print("=" * 65)
    print(f"  train_cologne_curriculum — {net_name} — {algo.upper()}")
    print(f"  reward={args.reward_type}  seed={args.seed}"
          f"  total_steps={args.total_steps}")
    print(f"  Curriculum stages:")
    for i, (frac, steps) in enumerate(stage_steps):
        print(f"    Stage {i+1}: {int(frac*100):3d}% demand  {steps:>7d} steps")
    print(f"  save_dir: {save_dir}")
    print("=" * 65)

    # ── Ensure route files per stage ──────────────────────────────────────────
    if args.rand:
        stage_routes = []
        for frac, _ in stage_steps:
            p = rand_route_path(route_file, frac)
            if not p.exists():
                sys.exit(
                    f"[ERR] Khong tim thay random route: {p}\n"
                    f"      Chay truoc: python make_random_routes.py "
                    f"--net {args.net_file} --rou {args.route_file}"
                )
            print(f"  [rand] {p.name}", flush=True)
            stage_routes.append(str(p))
    else:
        stage_routes = [
            str(ensure_scaled_route(route_file, frac,
                                    args.window_begin, args.window_end, args.seed))
            for frac, _ in stage_steps
        ]

    # ── Build initial model on Stage-1 env ────────────────────────────────────
    first_env = MultiAgentVecEnv(make_env_fn(
        net_file, stage_routes[0], args.reward_type,
        args.seed, args.sim_duration, args.sumo_begin, args.gui,
    ))
    n_agents = first_env.num_envs
    print(f"\n  n_agents={n_agents} | "
          f"obs={first_env.observation_space.shape} | "
          f"act={first_env.action_space}", flush=True)

    step_logger = StepLogger(
        log_freq=5_000, total_steps=args.total_steps, algo=algo)

    if algo in ("dqn", "ddqn"):
        scaled_buffer = args.buffer_size * n_agents
        print(f"  buffer_size={args.buffer_size} x {n_agents} agents = {scaled_buffer}")
        AlgoClass = DoubleDQN if algo == "ddqn" else DQN
        model = AlgoClass(
            "MlpPolicy", first_env,
            learning_rate          = lr,
            buffer_size            = scaled_buffer,
            learning_starts        = 5_000,
            batch_size             = 128,
            tau                    = 1.0,
            gamma                  = 0.99,
            train_freq             = (4, "step"),
            gradient_steps         = 1,
            target_update_interval = 1_000,
            exploration_fraction   = 0.25,   # extended for curriculum
            exploration_final_eps  = 0.05,
            optimize_memory_usage  = False,
            verbose                = 1,
            seed                   = args.seed,
            device                 = "cpu",
        )
    else:  # ppo
        model = PPO(
            "MlpPolicy", first_env,
            learning_rate = lr,
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

    current_env = first_env

    # ── Curriculum loop ───────────────────────────────────────────────────────
    cumulative = 0
    for stage_idx, ((frac, steps), route_str) in enumerate(
            zip(stage_steps, stage_routes)):
        pct = int(round(frac * 100))
        print(f"\n{'─'*60}", flush=True)
        print(f"  Stage {stage_idx+1}/4  {pct}% demand  {steps} steps"
              f"  (cumulative -> {cumulative+steps})", flush=True)
        print(f"  Route: {Path(route_str).name}", flush=True)
        print(f"{'─'*60}", flush=True)

        # Swap env at stage transitions
        if stage_idx > 0:
            current_env.close()
            new_env = MultiAgentVecEnv(make_env_fn(
                net_file, route_str, args.reward_type,
                args.seed, args.sim_duration, args.sumo_begin, args.gui,
            ))
            model.set_env(new_env)
            current_env = new_env

            if algo in ("dqn", "ddqn"):
                if args.clear_buffer:
                    model.replay_buffer.reset()
                    print(f"  Replay buffer cleared.", flush=True)
                # Restore some exploration at each harder stage
                model.exploration_rate = max(model.exploration_rate, 0.15)
                print(f"  eps -> {model.exploration_rate:.3f}", flush=True)

        ckpt_cb = CheckpointCallback(
            save_freq   = max(args.save_freq // n_agents, 1),
            save_path   = str(save_dir / "checkpoints"),
            name_prefix = f"{net_name}_{algo}_s{stage_idx+1}_ckpt",
        )

        model.learn(
            total_timesteps     = steps,
            callback            = CallbackList([ckpt_cb, step_logger]),
            reset_num_timesteps = (stage_idx == 0),
            progress_bar        = False,
            log_interval        = 1,
        )

        stage_path = save_dir / f"{net_name}_{algo}_stage{stage_idx+1}_{pct}pct"
        model.save(str(stage_path))
        print(f"  Stage {stage_idx+1} saved -> {stage_path.name}.zip", flush=True)

        cumulative += steps

    # ── Save final model (compatible with evaluate_cologne3_real.py) ──────────
    current_env.close()
    final_path = save_dir / f"{net_name}_{algo}_final_model"
    model.save(str(final_path))
    print(f"\n[{net_name}-{algo.upper()}] Curriculum done.")
    print(f"  Final model -> {final_path}.zip")
    print(f"  Total steps : {cumulative}")


if __name__ == "__main__":
    main()
