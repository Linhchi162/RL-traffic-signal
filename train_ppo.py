"""
train_ppo.py — Huấn luyện agent PPO điều khiển đèn tín hiệu

Hỗ trợ hai chế độ:
  --mode single : 1 nút giao, dùng file RLTSCQ gốc (caliberated_net.xml)
  --mode multi  : 4 nút giao (2x2 grid), parameter sharing — 1 policy PPO
                  điều khiển cả 4 đèn, mỗi đèn quan sát cục bộ.

Sử dụng:
    python train_ppo.py --save_dir ./output --mode single --total_steps 100000
    python train_ppo.py --save_dir ./out4   --mode multi  --total_steps 200000

    # Đa luồng (4 môi trường song song):
    python train_ppo.py --save_dir ./output --n_envs 4 --total_steps 100000

Tham số:
    --save_dir      : Thư mục lưu model và log (bắt buộc)
    --mode          : single | multi (mặc định: single)
    --reward_type   : queue | diff-waiting-time | average-speed (mặc định: queue)
    --total_steps   : Tổng số bước huấn luyện (mặc định: 100000)
    --save_freq     : Tần suất lưu checkpoint (mặc định: 10000)
    --seed          : Seed ngẫu nhiên (mặc định: 42)
    --obs_mode      : raw | compressed | baseline (mặc định: raw)
    --n_envs        : Số môi trường song song / SubprocVecEnv (mặc định: 1)
    --gui           : Bật giao diện SUMO GUI
"""

import argparse
import csv
import functools
import os
import sys
import time
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

if "SUMO_HOME" not in os.environ:
    sys.exit("[train_ppo] Loi: Chua khai bao bien moi truong 'SUMO_HOME'")

# Fix encoding tren Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_HERE = Path(__file__).parent

# --- Đường dẫn file SUMO ---
# Single: dùng mạng RLTSCQ gốc (1 nút giao đã calibrate)
_RLTSCQ_NETS = _HERE / "RLTSCQ" / "RLTSCQ-main" / "sumo_rl" / "nets" / "RLQ"
SINGLE_NET   = _RLTSCQ_NETS / "caliberated_net.xml"
SINGLE_ROUTE = _RLTSCQ_NETS / "train_flows.xml"   # fallback neu chua co generated

# Generated flows (TCCS 24:2018, da dang NS/WE): chon theo seed
_GENERATED_FLOWS = {
    42:  _HERE / "generated_flows" / "train_s0.xml",
    123: _HERE / "generated_flows" / "train_s1.xml",
    777: _HERE / "generated_flows" / "train_s2.xml",
}

# Multi: mạng 2x2 tự tạo (chạy sumo_nets/generate_net.py trước)
MULTI_NET   = _HERE / "sumo_nets" / "grid2x2.net.xml"
MULTI_ROUTE = _HERE / "sumo_nets" / "grid2x2_train.rou.xml"

from rl_controller.traffic_env import TrafficControlEnv
from rl_controller.state_builder import IntersectionStateExtractor, BaselineObservation
from rl_controller.encoder_obs import CompressedState_16D


# ---------------------------------------------------------------------------
# Factory môi trường — định nghĩa ở module-level để picklable trên Windows
# ---------------------------------------------------------------------------

def _make_env(net_file, route_file, obs_cls, reward_type, total_steps,
              seed, use_gui, is_multi):
    """Tạo một TrafficControlEnv (được gọi trong subprocess với SubprocVecEnv)."""
    env = TrafficControlEnv(
        net_file=net_file,
        route_file=route_file,
        csv_output="",
        use_gui=use_gui,
        sim_duration=total_steps + 5_000,
        single_agent=not is_multi,
        reward_fn=reward_type,
        obs_class=obs_cls,
        sumo_seed=seed,
    )
    if is_multi:
        return MultiAgentGymWrapper(env)
    return env


# ---------------------------------------------------------------------------
# Callback ghi log
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
            print(
                f"[{self.num_timesteps:>7,}/{self.total_steps:,}] "
                f"{pct:5.1f}% | "
                f"{elapsed/60:.1f}min elapsed | "
                f"~{remaining/60:.1f}min left",
                flush=True,
            )
            self._next_log += self.log_every
        return True


class TrainingMetricsCallback(BaseCallback):
    """
    Ghi các chỉ số huấn luyện PPO vào CSV sau mỗi rollout.

    Columns: timestep, loss, value_loss, entropy_loss, approx_kl,
             clip_fraction, explained_variance, policy_gradient_loss
    """

    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self._file = None
        self._writer = None

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
                "timestep": self.num_timesteps,
                "loss": vals.get("train/loss"),
                "value_loss": vals.get("train/value_loss"),
                "entropy_loss": vals.get("train/entropy_loss"),
                "approx_kl": vals.get("train/approx_kl"),
                "clip_fraction": vals.get("train/clip_fraction"),
                "explained_variance": vals.get("train/explained_variance"),
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
# Wrapper multi-agent → single gym.Env (Parameter Sharing)
# ---------------------------------------------------------------------------

class ParameterSharingWrapper:
    """
    Bọc TrafficControlEnv (multi-agent) thành giao diện gym.Env đơn.

    Chiến lược Parameter Sharing:
      - Mỗi bước, thu thập observation từ TẤT CẢ nút giao cần hành động.
      - Policy PPO (dùng chung) predict action cho TỪNG agent một.
      - Collect (obs, action, reward, done) từng cặp vào cùng replay buffer.
      - → 1 policy học từ kinh nghiệm của N nút giao cùng lúc.

    Chú ý: Không phải gym.Env thật sự, dùng train_multiagent() riêng.
    """

    def __init__(self, env: TrafficControlEnv):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space      = env.action_space
        self._pending_obs: dict = {}   # {signal_id: obs}

    def reset(self):
        obs_dict = self.env.reset()
        self._pending_obs = obs_dict
        # Trả về obs của agent đầu tiên (không quan trọng, warmup)
        first_obs = next(iter(obs_dict.values())) if obs_dict else \
            np.zeros(self.observation_space.shape, dtype=np.float32)
        return first_obs, {}

    @property
    def signal_ids(self):
        return self.env.signal_ids

    @property
    def sim_step(self):
        return self.env.sim_step


# ---------------------------------------------------------------------------
# Training loop cho parameter sharing
# ---------------------------------------------------------------------------

def train_multiagent(
    env: TrafficControlEnv,
    agent: PPO,
    total_steps: int,
    callbacks: list,
    log_interval: int = 1,
):
    """
    Vòng lặp huấn luyện PPO với parameter sharing cho multi-agent.

    Mỗi "timestep" trong PPO tương ứng với một (obs, action, reward) pair
    từ bất kỳ nút giao nào. Số timesteps thực = total_steps.

    Args:
        env: TrafficControlEnv với single_agent=False
        agent: PPO agent đã khởi tạo với env wrapper
        total_steps: Tổng timesteps thu thập
        callbacks: Danh sách callback SB3
        log_interval: In log mỗi N rollout
    """
    print(f"\n[Multi-Agent] Parameter Sharing — {len(env.signal_ids)} nút giao")
    print(f"[Multi-Agent] Tổng {total_steps} timesteps\n")
    agent.learn(
        total_timesteps=total_steps,
        callback=callbacks,
        log_interval=log_interval,
    )


# ---------------------------------------------------------------------------
# Tạo môi trường gym wrapper cho multi mode
# ---------------------------------------------------------------------------

class MultiAgentGymWrapper(gym.Env):
    """
    Chuyen doi TrafficControlEnv multi-agent thanh gym.Env chuan
    de dung voi stable_baselines3.

    Round-robin parameter sharing:
    moi step SB3 = 1 quyet dinh cua 1 agent.
    Policy dung chung cho tat ca agent.
    """

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
        # Neu queue con pending obs -> tra ve obs tiep theo ma khong tien sim
        if self._obs_queue:
            _sid, next_obs = self._obs_queue.pop(0)
            reward = float(np.mean(list(self._reward_accum.values()))) if self._reward_accum else 0.0
            return next_obs, reward, False, self._done, {}

        # Tien mo phong: gui action cho tat ca agent can hanh dong
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
        return obs, mean_reward, False, self._done, info

    def close(self):
        self._env.close()

    def render(self):
        return self._env.render()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Huấn luyện PPO điều khiển đèn tín hiệu (SUMO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--save_dir", required=True, help="Thư mục lưu kết quả")
    p.add_argument("--mode", default="single", choices=["single", "multi"],
                   help="single=1 nút giao (RLTSCQ) | multi=4 nút giao (2x2 grid)")
    p.add_argument("--reward_type", default="queue",
                   choices=["queue", "diff-waiting-time", "average-speed", "pressure", "wait-clip"],
                   help="Ham phan thuong: queue | diff-waiting-time | pressure | average-speed | wait-clip")
    p.add_argument("--total_steps", type=int, default=100_000,
                   help="Tổng số bước huấn luyện")
    p.add_argument("--save_freq", type=int, default=10_000,
                   help="Tần suất lưu checkpoint")
    p.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    p.add_argument("--obs_mode", default="raw",
                   choices=["raw", "compressed_4d", "compressed_8d", "compressed_16d", "compressed_19d", "compressed_32d", "baseline"],
                   help="raw=19D | compressed_Nd=autoencoder | baseline=unnormalized")
    p.add_argument("--n_envs", type=int, default=1,
                   help="Số môi trường song song (SubprocVecEnv). >1 tăng tốc thu thập mẫu. "
                        "Lưu ý: n_envs>1 yêu cầu đủ RAM/CPU; không dùng với --gui")
    p.add_argument("--gui", action="store_true", default=False,
                   help="Bật giao diện SUMO GUI")
    p.add_argument("--log_every", type=int, default=10_000,
                   help="In progress moi N steps (default: 10000)")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (default: 3e-4 for single, 3e-6 for multi)")
    return p.parse_args()


def main():
    args = parse_args()
    is_multi = (args.mode == "multi")
    os.makedirs(args.save_dir, exist_ok=True)

    # Kiểm tra file SUMO
    if is_multi:
        if not MULTI_NET.exists():
            sys.exit(
                f"[train_ppo] Chưa có file mạng 2x2!\n"
                f"Hãy chạy trước: python sumo_nets/generate_net.py"
            )
        net_file, route_file = str(MULTI_NET), str(MULTI_ROUTE)
    else:
        net_file = str(SINGLE_NET)
        generated = _GENERATED_FLOWS.get(args.seed)
        route_file = str(generated if generated and generated.exists() else SINGLE_ROUTE)

    # Chọn observation class
    if args.obs_mode == "compressed_4d":
        from rl_controller.encoder_obs import CompressedState_4D
        obs_cls = CompressedState_4D
    elif args.obs_mode == "compressed_8d":
        from rl_controller.encoder_obs import CompressedState_8D
        obs_cls = CompressedState_8D
    elif args.obs_mode == "compressed_16d":
        from rl_controller.encoder_obs import CompressedState_16D
        obs_cls = CompressedState_16D
    elif args.obs_mode == "compressed_19d":
        from rl_controller.encoder_obs import CompressedState_19D
        obs_cls = CompressedState_19D
    elif args.obs_mode == "compressed_32d":
        from rl_controller.encoder_obs import CompressedState_32D
        obs_cls = CompressedState_32D
    elif args.obs_mode == "baseline":
        from rl_controller.state_builder import BaselineObservation
        obs_cls = BaselineObservation
    else:
        from rl_controller.state_builder import IntersectionStateExtractor
        obs_cls = IntersectionStateExtractor

    # Đường dẫn output
    metrics_log    = os.path.join(args.save_dir, "training_metrics.csv")
    checkpoint_dir = os.path.join(args.save_dir, "checkpoints")
    env_csv        = os.path.join(args.save_dir, "env_metrics")
    final_model    = os.path.join(args.save_dir, "ppo_final_model.zip")

    n_envs = max(1, args.n_envs)
    if args.gui and n_envs > 1:
        print("[train_ppo] Cảnh báo: --gui không hỗ trợ n_envs>1, đặt lại n_envs=1")
        n_envs = 1

    print("=" * 62)
    print(f"  PPO --- {'4 nut giao (Parameter Sharing)' if is_multi else '1 nut giao'}")
    print(f"  Reward   : {args.reward_type}")
    print(f"  Obs mode : {args.obs_mode}")
    print(f"  Steps    : {args.total_steps:,}")
    print(f"  n_envs   : {n_envs}  {'(SubprocVecEnv)' if n_envs > 1 else '(single)'}")
    print(f"  Net file : {Path(net_file).name}")
    print(f"  Save dir : {args.save_dir}")
    print("=" * 62)

    # Tạo danh sách factory function — mỗi env có seed khác nhau
    env_fns = [
        functools.partial(
            _make_env,
            net_file, route_file, obs_cls, args.reward_type,
            args.total_steps, args.seed + i, args.gui, is_multi,
        )
        for i in range(n_envs)
    ]

    if n_envs > 1:
        # SubprocVecEnv trên Linux/Mac; Windows dùng start_method="spawn" có thể deadlock
        # nên fallback về DummyVecEnv (vẫn chạy n_envs env, nhưng tuần tự trong 1 process)
        if sys.platform == "win32":
            print(f"[train_ppo] Windows: dùng DummyVecEnv thay SubprocVecEnv ({n_envs} envs tuần tự)")
            train_env = DummyVecEnv(env_fns)
        else:
            train_env = SubprocVecEnv(env_fns, start_method="fork")
    else:
        train_env = DummyVecEnv(env_fns)

    metrics_cb  = TrainingMetricsCallback(log_path=metrics_log)
    progress_cb = ProgressCallback(total_steps=remaining, log_every=args.log_every)
    # save_freq đơn vị là "timesteps per env" với VecEnv
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, args.save_freq // n_envs),
        save_path=checkpoint_dir,
        name_prefix="ppo_ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    lr = args.lr if args.lr is not None else (3e-4 if args.mode == "single" else 3e-6)

    # Resume từ checkpoint mới nhất nếu có
    ckpt_files = sorted(Path(checkpoint_dir).glob("ppo_ckpt_*_steps.zip")) if Path(checkpoint_dir).exists() else []
    steps_done = 0
    if ckpt_files:
        latest_ckpt = ckpt_files[-1]
        try:
            steps_done = int(latest_ckpt.stem.split("_steps")[0].split("_")[-1])
        except ValueError:
            steps_done = 0
    remaining = args.total_steps - steps_done

    if ckpt_files and steps_done > 0 and remaining > 0:
        print(f"\nResume tu checkpoint: {latest_ckpt.name} (da chay {steps_done:,} buoc, con {remaining:,} buoc)")
        agent = PPO.load(str(latest_ckpt), env=train_env, learning_rate=lr, seed=args.seed)
    else:
        if steps_done >= args.total_steps:
            print(f"\nDa du {args.total_steps:,} buoc, luu model va thoat.")
            agent = PPO.load(str(ckpt_files[-1]), env=train_env, learning_rate=lr, seed=args.seed)
            agent.save(final_model)
            train_env.close()
            print("Hoan tat.")
            return
        agent = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=lr,
            n_steps=200,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            seed=args.seed,
        )

    print(f"\nBat dau huan luyen {remaining:,} buoc (tong {args.total_steps:,})...")
    agent.learn(
        total_timesteps=remaining,
        callback=[metrics_cb, checkpoint_cb, progress_cb],
        log_interval=10_000,
        reset_num_timesteps=(steps_done == 0),
    )

    print(f"\nHuấn luyện xong. Lưu model: {final_model}")
    agent.save(final_model)
    train_env.close()
    print("Hoàn tất.")


if __name__ == "__main__":
    main()
