"""
train_dqn.py — Huan luyen DQN / DDQN dieu khien den tin hieu

  - State space: RESCO per-lane features (active_phase, veh_count, total_wait, queue, speed_sum)
  - Reward: wait-clip | queue | pressure
  - Algo: dqn (vanilla) | ddqn (Double DQN)
  - Moi nut giao: 1 agent doc lap (khong parameter sharing)
  - Action: chon pha xanh tiep theo

Su dung:
    python train_dqn.py --save_dir ./out --algo ddqn --reward_type wait-clip --mode multi --seed 42
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Fix encoding Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch as th
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
import time
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

if "SUMO_HOME" not in os.environ:
    sys.exit("[train_dqn] SUMO_HOME chua duoc khai bao")

import traci
import sumolib

_HERE = Path(__file__).parent
_RLTSCQ_NETS = _HERE / "RLTSCQ" / "RLTSCQ-main" / "sumo_rl" / "nets" / "RLQ"

SINGLE_NET   = _RLTSCQ_NETS / "caliberated_net.xml"
SINGLE_ROUTE = _RLTSCQ_NETS / "train_flows.xml"   # fallback neu chua co generated

# Generated flows (TCCS 24:2018, da dang NS/WE): chon theo seed
_GENERATED_FLOWS = {
    42:  _HERE / "generated_flows" / "train_s0.xml",
    123: _HERE / "generated_flows" / "train_s1.xml",
    777: _HERE / "generated_flows" / "train_s2.xml",
}
MULTI_NET    = _HERE / "sumo_nets" / "grid2x2.net.xml"
MULTI_ROUTE  = _HERE / "sumo_nets" / "grid2x2_train.rou.xml"

# RESCO reward clipping parameters (Eq. 19)
REWARD_ALPHA = 100.0   # normalization factor
REWARD_MIN   = -5.0    # Rmin
REWARD_MAX   = 0.0     # Rmax

USE_LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ

QUEUE_NORM = 30.0   # xe toi da gia dinh moi nut giao de chuan hoa queue reward


# ---------------------------------------------------------------------------
# Double DQN
# ---------------------------------------------------------------------------

class DoubleDQN(DQN):
    """Double DQN: dung online network de CHON action, target network de DANH GIA."""

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # Double DQN: online net chon action, target net danh gia
                best_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                next_q_values = self.q_net_target(replay_data.next_observations).gather(1, best_actions)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


# ---------------------------------------------------------------------------
# RESCO Observation Builder
# ---------------------------------------------------------------------------

class RescoObservation:
    """
    Xay dung vector quan sat theo dac ta RESCO:
    Voi moi lane: [active_phase, vehicle_count, total_wait, queue_length, speed_sum]
    Ket qua flatten thanh 1 vector phang.

    Theo bai bao: "state space defined as a matrix of per-lane features
    including an active phase indicator, approaching vehicle count,
    total wait time, queue length, and the sum of vehicle speeds."
    """

    N_FEATURES_PER_LANE = 5  # active_phase, veh_count, total_wait, queue, speed_sum

    def __init__(self, ts_id: str, sumo_conn, n_lanes: Optional[int] = None):
        self.ts_id = ts_id
        self.sumo = sumo_conn

        # Lay danh sach lanes
        self.lanes = list(dict.fromkeys(
            self.sumo.trafficlight.getControlledLanes(self.ts_id)
        ))
        self.n_lanes = len(self.lanes)

    @property
    def obs_dim(self) -> int:
        return self.n_lanes * self.N_FEATURES_PER_LANE

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    def __call__(self) -> np.ndarray:
        """Tinh toan va tra ve vector quan sat hien tai."""
        try:
            cur_phase = self.sumo.trafficlight.getPhase(self.ts_id)
            logics = self.sumo.trafficlight.getAllProgramLogics(self.ts_id)
            if logics:
                state_str = logics[0].phases[cur_phase].state if cur_phase < len(logics[0].phases) else ""
            else:
                state_str = ""

            # Phat hien lane duoc phuc vu trong pha hien tai
            active_lanes = set()
            links = self.sumo.trafficlight.getControlledLinks(self.ts_id)
            for link in links:
                if not (isinstance(link, (list, tuple)) and len(link) >= 2):
                    continue
                inner = link[0]
                if not (isinstance(inner, (list, tuple)) and len(inner) >= 1):
                    continue
                in_lane = inner[0]
                link_pos = link[1]
                if link_pos < len(state_str) and state_str[link_pos].upper() == "G":
                    active_lanes.add(in_lane)
        except Exception:
            active_lanes = set()

        feats = []
        for lane in self.lanes:
            try:
                vehs = self.sumo.lane.getLastStepVehicleIDs(lane)
                veh_count = len(vehs)
                queue = self.sumo.lane.getLastStepHaltingNumber(lane)
                total_wait = sum(
                    self.sumo.vehicle.getWaitingTime(v) for v in vehs
                )
                speed_sum = sum(
                    self.sumo.vehicle.getSpeed(v) for v in vehs
                )
                active = 1.0 if lane in active_lanes else 0.0
            except Exception:
                veh_count = 0
                queue = 0
                total_wait = 0.0
                speed_sum = 0.0
                active = 0.0

            feats.extend([active, float(veh_count), total_wait / 100.0,
                          float(queue), speed_sum / 10.0])

        return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# RESCO Reward Function
# ---------------------------------------------------------------------------

def resco_reward(ts_id: str, sumo_conn) -> float:
    """
    Tinh phan thuong RESCO theo Eq. 19:
        Rt = clip(-sum(wait_l) / alpha, Rmin, Rmax)

    Args:
        ts_id:     ID den tin hieu.
        sumo_conn: Ket noi TraCI.

    Returns:
        Phan thuong da clip.
    """
    try:
        lanes = list(dict.fromkeys(
            sumo_conn.trafficlight.getControlledLanes(ts_id)
        ))
        total_wait = 0.0
        for lane in lanes:
            vehs = sumo_conn.lane.getLastStepVehicleIDs(lane)
            for v in vehs:
                total_wait += sumo_conn.vehicle.getWaitingTime(v)
    except Exception:
        total_wait = 0.0

    raw = -total_wait / REWARD_ALPHA
    return float(max(REWARD_MIN, min(REWARD_MAX, raw)))


def queue_reward(ts_id: str, sumo_conn) -> float:
    """Phan thuong dua tren tong xe dung (hang cho), clip [-5, 0]."""
    try:
        lanes = list(dict.fromkeys(sumo_conn.trafficlight.getControlledLanes(ts_id)))
        total_halting = sum(sumo_conn.lane.getLastStepHaltingNumber(l) for l in lanes)
    except Exception:
        total_halting = 0
    return float(max(REWARD_MIN, min(REWARD_MAX, -total_halting / QUEUE_NORM)))


def pressure_reward(ts_id: str, sumo_conn) -> float:
    """Phan thuong ap luc: clip(incoming_halting - outgoing_halting, -5, 5)."""
    try:
        links = sumo_conn.trafficlight.getControlledLinks(ts_id)
        incoming, outgoing = set(), set()
        for link in links:
            if isinstance(link, (list, tuple)) and len(link) >= 1:
                inner = link[0]
                if isinstance(inner, (list, tuple)) and len(inner) >= 2:
                    incoming.add(inner[0])
                    outgoing.add(inner[1])
        in_q  = sum(sumo_conn.lane.getLastStepHaltingNumber(l) for l in incoming)
        out_q = sum(sumo_conn.lane.getLastStepHaltingNumber(l) for l in outgoing)
        pressure = (in_q - out_q) / max(QUEUE_NORM, 1.0)
    except Exception:
        pressure = 0.0
    return float(max(-5.0, min(5.0, -pressure)))


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


REWARD_FN_MAP = {
    "wait-clip": resco_reward,
    "queue":     queue_reward,
    "pressure":  pressure_reward,
}


# ---------------------------------------------------------------------------
# RESCO DQN Environment (Single Intersection)
# ---------------------------------------------------------------------------

class RescoDQNEnv(gym.Env):
    """
    Moi truong RESCO-DQN cho 1 nut giao.

    Action space: chon pha xanh tiep theo (Discrete(n_green_phases)).
    Observation: vector dac trung per-lane.
    Reward: clip(-total_wait / alpha, Rmin, Rmax).
    """

    metadata = {"render_modes": []}
    _conn_counter = 0

    def __init__(
        self,
        net_file: str,
        route_file: str,
        ts_index: int = 0,
        sim_duration: int = 20_000,
        step_interval: int = 5,
        amber_sec: int = 5,
        sumo_seed: int = 42,
        use_gui: bool = False,
        reward_type: str = "wait-clip",
    ):
        super().__init__()
        self._net = net_file
        self._route = route_file
        self._ts_index = ts_index
        self._sim_duration = sim_duration
        self._step_interval = step_interval
        self._amber_sec = amber_sec
        self._sumo_seed = sumo_seed
        self._use_gui = use_gui
        self._reward_fn = REWARD_FN_MAP.get(reward_type, resco_reward)
        self._label = f"dqn_{RescoDQNEnv._conn_counter}"
        RescoDQNEnv._conn_counter += 1
        self.sumo = None

        # Lay thong tin tu ket noi tam
        if USE_LIBSUMO:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net])
            tmp = traci
        else:
            traci.start(
                [sumolib.checkBinary("sumo"), "-n", self._net],
                label=f"init_{self._label}",
            )
            tmp = traci.getConnection(f"init_{self._label}")

        signal_ids = list(tmp.trafficlight.getIDList())
        self._ts_id = signal_ids[ts_index % len(signal_ids)]

        # Phat hien pha xanh
        self._green_phases = []
        logics = tmp.trafficlight.getAllProgramLogics(self._ts_id)
        if logics:
            for i, ph in enumerate(logics[0].phases):
                if "G" in ph.state.upper() and "Y" not in ph.state.upper():
                    self._green_phases.append(i)
        if not self._green_phases:
            self._green_phases = [0, 2, 4, 6][:len(logics[0].phases) // 2]

        # Khoi tao obs builder tam de lay spaces
        self.sumo = tmp
        self._obs_builder = RescoObservation(self._ts_id, tmp)
        _tmp_obs_space = self._obs_builder.observation_space()
        tmp.close()
        self.sumo = None

        self.observation_space = _tmp_obs_space
        self.action_space = spaces.Discrete(len(self._green_phases))
        self._episode = 0

    def reset(self, seed=None, **kwargs):
        if self.sumo is not None:
            self._close_sumo()
        self._episode += 1

        binary = "sumo-gui" if self._use_gui else "sumo"
        cmd = [
            sumolib.checkBinary(binary),
            "-n", self._net,
            "-r", self._route,
            "--time-to-teleport", "-1",
            "--no-warnings",
        ]
        if self._sumo_seed == "random":
            cmd.append("--random")
        else:
            cmd.extend(["--seed", str(self._sumo_seed)])

        if USE_LIBSUMO:
            traci.start(cmd)
            self.sumo = traci
        else:
            traci.start(cmd, label=self._label)
            self.sumo = traci.getConnection(self._label)

        self._obs_builder = RescoObservation(self._ts_id, self.sumo)
        self._sim_time = 0.0
        self._current_green_phase_idx = 0
        self._phase_timer = 0

        obs = self._obs_builder()
        return obs, {}

    def step(self, action: int):
        """
        Ap dung pha xanh duoc chon va tien mo phong step_interval giay.

        Args:
            action: Chi so trong green_phases.
        """
        target_phase = self._green_phases[action % len(self._green_phases)]

        # Neu can chuyen pha, qua pha vang truoc
        current = self.sumo.trafficlight.getPhase(self._ts_id)
        if current != target_phase:
            # Pha vang la pha le sau pha xanh le truoc
            amber_phase = current + 1 if (current + 1) < len(
                self.sumo.trafficlight.getAllProgramLogics(self._ts_id)[0].phases
            ) else 1
            self.sumo.trafficlight.setPhase(self._ts_id, amber_phase)
            for _ in range(self._amber_sec):
                self.sumo.simulationStep()
                self._sim_time += 1
            self.sumo.trafficlight.setPhase(self._ts_id, target_phase)

        # Tien mo phong step_interval giay
        for _ in range(self._step_interval):
            self.sumo.simulationStep()
            self._sim_time += 1

        obs = self._obs_builder()
        reward = self._reward_fn(self._ts_id, self.sumo)
        truncated = self._sim_time >= self._sim_duration
        return obs, reward, False, truncated, {}

    def close(self):
        self._close_sumo()

    def _close_sumo(self):
        if self.sumo is None:
            return
        try:
            if not USE_LIBSUMO:
                traci.switch(self._label)
            traci.close()
        except Exception:
            pass
        self.sumo = None

    def __del__(self):
        self._close_sumo()


# ---------------------------------------------------------------------------
# Multi-intersection Round-Robin DQN Wrapper
# ---------------------------------------------------------------------------

class MultiDQNWrapper(gym.Env):
    """
    Boc nhieu RescoDQNEnv thanh 1 gym.Env de dung voi SB3 DQN.
    Dung round-robin: moi step = 1 quyet dinh cua 1 nut giao.
    """

    metadata = {"render_modes": []}

    def __init__(self, envs: List[RescoDQNEnv]):
        super().__init__()
        self._envs = envs
        self._n = len(envs)
        self._ptr = 0
        self._obs_cache: List[Optional[np.ndarray]] = [None] * self._n
        self._done = False

        # Tat ca envs dung chung obs/action space
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space

    @property
    def unwrapped(self):
        return self._envs[0]

    def reset(self, seed=None, **kwargs):
        obs_list = []
        for env in self._envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        self._obs_cache = obs_list
        self._ptr = 0
        self._done = False
        return self._obs_cache[self._ptr], {}

    def step(self, action: int):
        env = self._envs[self._ptr]
        obs, reward, _, truncated, info = env.step(action)
        self._obs_cache[self._ptr] = obs

        if truncated:
            self._done = True

        self._ptr = (self._ptr + 1) % self._n
        next_obs = self._obs_cache[self._ptr]
        if next_obs is None:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return next_obs, reward, False, self._done, info

    def close(self):
        for env in self._envs:
            env.close()

    def render(self):
        pass


# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Huan luyen DQN/DDQN dieu khien den tin hieu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--save_dir", required=True)
    p.add_argument("--algo", default="dqn", choices=["dqn", "ddqn"],
                   help="Thuat toan: dqn (vanilla) | ddqn (Double DQN)")
    p.add_argument("--reward_type", default="wait-clip",
                   choices=["wait-clip", "queue", "pressure"],
                   help="Ham phan thuong: wait-clip | queue | pressure")
    p.add_argument("--mode", default="multi", choices=["single", "multi"])
    p.add_argument("--total_steps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_freq", type=int, default=10_000)
    p.add_argument("--log_every", type=int, default=10_000,
                   help="In progress moi N steps (default: 10000)")
    p.add_argument("--gui", action="store_true", default=False)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    is_multi = (args.mode == "multi")
    os.makedirs(args.save_dir, exist_ok=True)

    net_file = str(MULTI_NET if is_multi else SINGLE_NET)
    if is_multi:
        route_file = str(MULTI_ROUTE)
    else:
        generated = _GENERATED_FLOWS.get(args.seed)
        route_file = str(generated if generated and generated.exists() else SINGLE_ROUTE)

    AlgoClass = DoubleDQN if args.algo == "ddqn" else DQN
    algo_label = args.algo.upper()

    print("=" * 55)
    print(f"  Huan luyen {algo_label} | {'4 nut' if is_multi else '1 nut'}")
    print(f"  Reward  : {args.reward_type}")
    print(f"  Steps   : {args.total_steps:,} | Seed: {args.seed}")
    print(f"  Net     : {Path(net_file).name}")
    print("=" * 55)

    # Khoi tao moi truong
    env_kwargs = dict(
        net_file=net_file,
        route_file=route_file,
        sim_duration=args.total_steps + 5_000,
        sumo_seed=args.seed,
        use_gui=args.gui,
        reward_type=args.reward_type,
    )
    if is_multi:
        envs = [RescoDQNEnv(ts_index=i, **env_kwargs) for i in range(4)]
        train_env = MultiDQNWrapper(envs)
    else:
        train_env = RescoDQNEnv(ts_index=0, **env_kwargs)

    checkpoint_dir = os.path.join(args.save_dir, "checkpoints")
    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=checkpoint_dir,
        name_prefix="dqn_ckpt",
    )

    agent = AlgoClass(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        verbose=1,
        seed=args.seed,
    )

    progress_cb = ProgressCallback(total_steps=args.total_steps, log_every=args.log_every)

    print(f"\nBat dau huan luyen {args.total_steps:,} buoc...")
    agent.learn(
        total_timesteps=args.total_steps,
        callback=[checkpoint_cb, progress_cb],
        log_interval=10_000,
    )

    final_model = os.path.join(args.save_dir, "dqn_final_model.zip")
    agent.save(final_model)
    train_env.close()
    print(f"\nHoan tat. Model: {final_model}")


if __name__ == "__main__":
    main()
