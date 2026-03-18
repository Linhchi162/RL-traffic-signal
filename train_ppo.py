"""
train_ppo.py — Huấn luyện agent PPO điều khiển đèn tín hiệu (1 nút giao)

Sử dụng:
    python train_ppo.py --save_dir ./output --total_steps 100000

Tham số:
    --save_dir      : Thư mục lưu model và log (bắt buộc)
    --reward_type   : queue | diff-waiting-time | average-speed (mặc định: queue)
    --total_steps   : Tổng số bước huấn luyện (mặc định: 100000)
    --save_freq     : Tần suất lưu checkpoint (mặc định: 10000)
    --seed          : Seed ngẫu nhiên (mặc định: 42)
    --obs_mode      : raw | compressed | baseline (mặc định: raw)
    --n_envs        : Số môi trường song song / DummyVecEnv (mặc định: 1)
    --gui           : Bật giao diện SUMO GUI
"""

import argparse
import csv
import functools
import os
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

if "SUMO_HOME" not in os.environ:
    sys.exit("[train_ppo] Loi: Chua khai bao bien moi truong 'SUMO_HOME'")

# Fix encoding tren Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_HERE = Path(__file__).parent

_RLTSCQ_NETS = _HERE / "RLTSCQ" / "RLTSCQ-main" / "sumo_rl" / "nets" / "RLQ"
SINGLE_NET   = _RLTSCQ_NETS / "caliberated_net.xml"
SINGLE_ROUTE = _RLTSCQ_NETS / "train_flows.xml"   # fallback neu chua co generated

_GENERATED_FLOWS = {
    42:  _HERE / "generated_flows" / "train_s0.xml",
    123: _HERE / "generated_flows" / "train_s1.xml",
    777: _HERE / "generated_flows" / "train_s2.xml",
}

from rl_controller.traffic_env import TrafficControlEnv
from rl_controller.state_builder import IntersectionStateExtractor, BaselineObservation
from rl_controller.encoder_obs import CompressedState_16D


# ---------------------------------------------------------------------------
# Factory môi trường — định nghĩa ở module-level để picklable trên Windows
# ---------------------------------------------------------------------------

def _make_env(net_file, route_file, obs_cls, reward_type, sim_duration, seed, use_gui, amber_sec=0):
    """Tạo một TrafficControlEnv bọc bởi Monitor (để SB3 theo dõi ep_info_buffer)."""
    env = TrafficControlEnv(
        net_file=net_file,
        route_file=route_file,
        csv_output="",
        use_gui=use_gui,
        sim_duration=sim_duration,
        single_agent=True,
        reward_fn=reward_type,
        obs_class=obs_cls,
        sumo_seed=seed,
        amber_sec=amber_sec,
    )
    return Monitor(env)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class ProgressCallback(BaseCallback):
    def __init__(self, total_steps: int, log_every: int = 10_000):
        super().__init__()
        self.total_steps = total_steps
        self.log_every   = log_every
        self._next_log   = log_every
        self._t0         = None

    def _on_training_start(self):
        self._t0 = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_log:
            elapsed   = time.time() - self._t0
            pct       = self.num_timesteps / self.total_steps * 100
            rate      = self.num_timesteps / elapsed if elapsed > 0 else 1
            remaining = (self.total_steps - self.num_timesteps) / rate

            vals        = self.model.logger.name_to_value
            ep_buf      = getattr(self.model, "ep_info_buffer", [])
            rew         = float(np.mean([ep["r"] for ep in ep_buf])) if ep_buf else float("nan")
            ep_len      = float(np.mean([ep["l"] for ep in ep_buf])) if ep_buf else float("nan")
            expl_var    = vals.get("train/explained_variance", float("nan"))
            approx_kl   = vals.get("train/approx_kl", float("nan"))

            msg = (
                f"[{self.num_timesteps:>7,}/{self.total_steps:,}] "
                f"{pct:5.1f}% | "
                f"{elapsed/60:.1f}min | ~{remaining/60:.1f}min left | "
                f"rew={rew:+.3f} | expl_var={expl_var:.3f} | kl={approx_kl:.4f} | ep_len={ep_len:.0f}"
            )
            print(msg, flush=True)
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
            self._next_log += self.log_every
        return True


class TrainingMetricsCallback(BaseCallback):
    """Ghi chỉ số PPO vào CSV sau mỗi rollout."""

    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self._file    = None
        self._writer  = None

    def _on_training_start(self):
        self._file = open(self.log_path, "w", newline="")
        fields = [
            "timestep", "loss", "value_loss", "entropy_loss",
            "approx_kl", "clip_fraction", "explained_variance",
            "policy_gradient_loss",
        ]
        self._writer = csv.DictWriter(self._file, fieldnames=fields)
        self._writer.writeheader()

    def _on_rollout_end(self):
        try:
            vals = self.model.logger.name_to_value
            row = {
                "timestep":             self.num_timesteps,
                "loss":                 vals.get("train/loss"),
                "value_loss":           vals.get("train/value_loss"),
                "entropy_loss":         vals.get("train/entropy_loss"),
                "approx_kl":            vals.get("train/approx_kl"),
                "clip_fraction":        vals.get("train/clip_fraction"),
                "explained_variance":   vals.get("train/explained_variance"),
                "policy_gradient_loss": vals.get("train/policy_gradient_loss"),
            }
            self._writer.writerow(row)
            self._file.flush()
        except Exception as exc:
            print(f"[TrainingMetricsCallback] Lỗi ghi log: {exc}")

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self):
        if self._file:
            self._file.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Huấn luyện PPO điều khiển đèn tín hiệu (SUMO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--save_dir", required=True, help="Thư mục lưu kết quả")
    p.add_argument("--reward_type", default="queue",
                   choices=["queue", "diff-waiting-time", "average-speed", "pressure", "wait-clip"],
                   help="Ham phan thuong")
    p.add_argument("--total_steps", type=int, default=100_000)
    p.add_argument("--save_freq",   type=int, default=10_000)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--obs_mode", default="raw",
                   choices=["raw", "compressed_4d", "compressed_8d", "compressed_16d",
                            "compressed_19d", "compressed_32d", "baseline"])
    p.add_argument("--n_envs", type=int, default=1,
                   help="Số môi trường song song. Lưu ý: không dùng với --gui")
    p.add_argument("--sim_duration", type=int, default=None,
                   help="Độ dài mỗi episode (giây SUMO). Mặc định: total_steps + 5000")
    p.add_argument("--amber_sec",    type=int, default=0,
                   help="Thời gian đèn vàng (giây). 0 = đổi pha tức thì (khuyến nghị)")
    p.add_argument("--gui",      action="store_true", default=False)
    p.add_argument("--log_every", type=int, default=10_000)
    p.add_argument("--lr",        type=float, default=3e-4)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    net_file  = str(SINGLE_NET)
    generated = _GENERATED_FLOWS.get(args.seed)
    route_file = str(generated if generated and generated.exists() else SINGLE_ROUTE)

    obs_map = {
        "compressed_4d":  "rl_controller.encoder_obs.CompressedState_4D",
        "compressed_8d":  "rl_controller.encoder_obs.CompressedState_8D",
        "compressed_16d": "rl_controller.encoder_obs.CompressedState_16D",
        "compressed_19d": "rl_controller.encoder_obs.CompressedState_19D",
        "compressed_32d": "rl_controller.encoder_obs.CompressedState_32D",
    }
    if args.obs_mode in obs_map:
        module, cls_name = obs_map[args.obs_mode].rsplit(".", 1)
        import importlib
        obs_cls = getattr(importlib.import_module(module), cls_name)
    elif args.obs_mode == "baseline":
        obs_cls = BaselineObservation
    else:
        obs_cls = IntersectionStateExtractor

    metrics_log    = os.path.join(args.save_dir, "training_metrics.csv")
    checkpoint_dir = os.path.join(args.save_dir, "checkpoints")
    final_model    = os.path.join(args.save_dir, "ppo_final_model.zip")

    sim_dur = args.sim_duration if args.sim_duration else args.total_steps + 5_000
    n_envs = max(1, args.n_envs)
    if args.gui and n_envs > 1:
        print("[train_ppo] Cảnh báo: --gui không hỗ trợ n_envs>1, đặt lại n_envs=1")
        n_envs = 1

    print("=" * 55)
    print(f"  PPO — 1 nút giao")
    print(f"  Reward   : {args.reward_type}")
    print(f"  Obs mode : {args.obs_mode}")
    print(f"  Steps    : {args.total_steps:,}")
    print(f"  n_envs   : {n_envs}")
    print(f"  Sim dur  : {sim_dur:,}")
    print(f"  Amber sec: {args.amber_sec}")
    print(f"  Net file : {Path(net_file).name}")
    print(f"  Route    : {Path(route_file).name}")
    print(f"  Save dir : {args.save_dir}")
    print("=" * 55)

    env_fns = [
        functools.partial(
            _make_env,
            net_file, route_file, obs_cls, args.reward_type,
            sim_dur, args.seed + i, args.gui, args.amber_sec,
        )
        for i in range(n_envs)
    ]

    if n_envs > 1:
        if sys.platform == "win32":
            print(f"[train_ppo] Windows: dùng DummyVecEnv ({n_envs} envs tuần tự)")
            train_env = DummyVecEnv(env_fns)
        else:
            # spawn thay vi fork: tranh conflict libsumo trong child process
            train_env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        train_env = DummyVecEnv(env_fns)

    metrics_cb    = TrainingMetricsCallback(log_path=metrics_log)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, args.save_freq // n_envs),
        save_path=checkpoint_dir,
        name_prefix="ppo_ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    ckpt_files = sorted(Path(checkpoint_dir).glob("ppo_ckpt_*_steps.zip")) if Path(checkpoint_dir).exists() else []
    steps_done = 0
    if ckpt_files:
        try:
            steps_done = int(ckpt_files[-1].stem.split("_steps")[0].split("_")[-1])
        except ValueError:
            steps_done = 0
    remaining   = args.total_steps - steps_done
    progress_cb = ProgressCallback(total_steps=remaining, log_every=args.log_every)

    if ckpt_files and steps_done > 0 and remaining > 0:
        print(f"\nResume tu checkpoint: {ckpt_files[-1].name} (da chay {steps_done:,} buoc, con {remaining:,} buoc)")
        agent = PPO.load(str(ckpt_files[-1]), env=train_env, learning_rate=args.lr, seed=args.seed)
    else:
        if steps_done >= args.total_steps:
            print(f"\nDa du {args.total_steps:,} buoc, luu model va thoat.")
            agent = PPO.load(str(ckpt_files[-1]), env=train_env, learning_rate=args.lr, seed=args.seed)
            agent.save(final_model)
            train_env.close()
            print("Hoan tat.")
            return
        agent = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=args.lr,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.95,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            seed=args.seed,
        )

    print(f"\nBat dau huan luyen {remaining:,} buoc (tong {args.total_steps:,})...")
    agent.learn(
        total_timesteps=remaining,
        callback=[metrics_cb, checkpoint_cb, progress_cb],
        log_interval=4,
        reset_num_timesteps=(steps_done == 0),
    )

    print(f"\nHuấn luyện xong. Lưu model: {final_model}")
    agent.save(final_model)
    train_env.close()
    print("Hoàn tất.")


if __name__ == "__main__":
    main()
