"""
traffic_env.py — TrafficControlEnv

Môi trường Gymnasium cho bài toán điều khiển đèn tín hiệu giao thông
sử dụng simulator SUMO thông qua giao thức TraCI.

Hỗ trợ cả chế độ single-agent (trả về gym.Env chuẩn) và
multi-agent (trả về dict observations/rewards).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.utils import EzPickle

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    raise ImportError("Chưa khai báo biến môi trường 'SUMO_HOME'")

import sumolib
import traci

from .intersection import SignalController
from .state_builder import IntersectionStateExtractor

# Dùng libsumo nếu biến môi trường được đặt (nhanh hơn ~8x nhưng không có GUI)
USE_LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


class TrafficControlEnv(gym.Env):
    """
    Môi trường Gymnasium mô phỏng giao lộ đèn tín hiệu với SUMO.

    Mỗi bước, agent chọn thời gian xanh cho pha hiện tại.
    Phần thưởng mặc định dựa trên độ dài hàng chờ.

    Args:
        net_file: Đường dẫn tới file SUMO .net.xml.
        route_file: Đường dẫn tới file .rou.xml (lưu lượng xe).
        csv_output: Đường dẫn lưu kết quả CSV (None = không lưu).
        use_gui: Chạy với giao diện SUMO GUI.
        sim_duration: Tổng thời gian mô phỏng (giây).
        max_depart_delay: Thời gian tối đa chờ xe vào mô phỏng (-1 = không giới hạn).
        wait_memory: Số giây ghi nhớ thời gian chờ xe.
        teleport_time: Thời gian để dịch chuyển xe kẹt (-1 = không dịch chuyển).
        step_interval: Khoảng cách giữa hai quyết định của agent (giây).
        amber_sec: Thời gian đèn vàng (giây).
        min_green: Thời gian xanh tối thiểu (giây).
        max_green: Thời gian xanh tối đa (giây).
        single_agent: True = trả về gym.Env chuẩn; False = multi-agent dict.
        reward_fn: Hàm phần thưởng ("queue", "diff-waiting-time", ...).
        obs_class: Lớp observation function.
        sumo_seed: Seed ngẫu nhiên cho SUMO.
        fixed_signal: Nếu True, đèn chạy theo lịch cố định (không RL).
        show_warnings: In cảnh báo từ SUMO.
        extra_sumo_args: Tham số dòng lệnh bổ sung cho SUMO.
        render_mode: Chế độ render ('human' hoặc 'rgb_array').
    """

    metadata = {"render_modes": ["human", "rgb_array"]}
    _conn_counter = 0  # Đếm kết nối TraCI để tránh xung đột

    def __init__(
        self,
        net_file: str,
        route_file: str,
        csv_output: Optional[str] = None,
        use_gui: bool = False,
        sim_duration: int = 20_000,
        max_depart_delay: int = -1,
        wait_memory: int = 1000,
        teleport_time: int = -1,
        step_interval: int = 5,
        amber_sec: int = 5,
        min_green: int = 10,
        max_green: int = 40,
        enforce_max_green: bool = True,
        single_agent: bool = False,
        reward_fn: Union[str, Callable, dict, List] = "queue",
        reward_weights: Optional[List[float]] = None,
        obs_class=IntersectionStateExtractor,
        sumo_seed: Union[str, int] = "random",
        fixed_signal: bool = False,
        show_warnings: bool = True,
        extra_sumo_args: Optional[str] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        assert max_green > min_green, "max_green phải lớn hơn min_green"

        self.render_mode = render_mode
        self._net_path = net_file
        self._route_path = route_file
        self.use_gui = use_gui
        self._vdisplay = None

        self.sim_max_time = sim_duration
        self.step_interval = step_interval
        self.max_depart_delay = max_depart_delay
        self.wait_memory = wait_memory
        self.teleport_time = teleport_time
        self.amber_sec = amber_sec
        self.min_green = min_green
        self.max_green = max_green
        self.enforce_max_green = enforce_max_green
        self.single_agent = single_agent
        self.reward_fn = reward_fn
        self.reward_weights = reward_weights
        self.obs_class = obs_class
        self.sumo_seed = sumo_seed
        self.fixed_signal = fixed_signal
        self.show_warnings = show_warnings
        self.extra_sumo_args = extra_sumo_args
        self.csv_output = csv_output

        self._label = str(TrafficControlEnv._conn_counter)
        TrafficControlEnv._conn_counter += 1
        self.sumo: Optional[traci.Connection] = None
        self.controllers: Dict[str, SignalController] = {}

        # Kết nối tạm để đọc danh sách đèn và thiết lập không gian quan sát/hành động
        _init_cmd = [
            sumolib.checkBinary("sumo"),
            "-n", self._net_path,
            "-r", self._route_path,
            "--end", "1",
            "--no-step-log",
        ]
        if USE_LIBSUMO:
            traci.start(_init_cmd)
            tmp_conn = traci
        else:
            traci.start(_init_cmd, label="init_" + self._label)
            tmp_conn = traci.getConnection("init_" + self._label)

        self.signal_ids: List[str] = list(tmp_conn.trafficlight.getIDList())

        # Pre-compute observation/action space từ kết nối tạm
        self._action_space, self._obs_space = self._read_spaces_from_conn(tmp_conn)
        tmp_conn.close()

        # Thống kê mô phỏng
        self.vehicle_registry: Dict[str, dict] = {}
        self.reward_range = (-float("inf"), float("inf"))
        self.episode_count = 0
        self.step_metrics: List[dict] = []
        self.signal_states: Dict[str, Optional[np.ndarray]] = {}
        self.signal_rewards: Dict[str, Optional[float]] = {}

        self._arrived = 0
        self._departed = 0
        self._teleported = 0

    def _read_spaces_from_conn(self, conn):
        """
        Đọc observation space và action space từ kết nối TraCI tạm.

        Tạo một SignalController đuợc rút gọn (chỉ lấy spaces),
        không thực hiện simulation.
        """
        from gymnasium import spaces as gym_spaces
        import math

        action_space = gym_spaces.Discrete(self.max_green - self.min_green + 1)

        if not self.signal_ids:
            fallback = gym_spaces.Box(
                low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
            )
            return action_space, fallback

        # Tạo temporary SignalController để lấy observation space
        try:
            # Thiết lập env tạm để SignalController có thể tham chiếu
            self.sumo = conn
            self._build_controllers(conn)
            obs_space = self.controllers[self.signal_ids[0]].observation_space
            # Reset lại — controllers sẽ được xây lại khi reset()
            self.controllers = {}
            self.sumo = None
        except Exception as exc:
            print(f"[TrafficControlEnv] Cảnh báo: không đọc được obs space từ SUMO: {exc}")
            obs_space = gym_spaces.Box(
                low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
            )

        return action_space, obs_space

    # ------------------------------------------------------------------
    # Khởi động / kết thúc SUMO
    # ------------------------------------------------------------------

    def _launch_sumo(self):
        """Khởi động tiến trình SUMO và thiết lập kết nối TraCI."""
        binary = "sumo-gui" if (self.use_gui or self.render_mode) else "sumo"
        cmd = [
            sumolib.checkBinary(binary),
            "-n", self._net_path,
            "-r", self._route_path,
            "--max-depart-delay", str(self.max_depart_delay),
            "--waiting-time-memory", str(self.wait_memory),
            "--time-to-teleport", str(self.teleport_time),
        ]
        if self.sumo_seed == "random":
            cmd.append("--random")
        else:
            cmd.extend(["--seed", str(self.sumo_seed)])
        if not self.show_warnings:
            cmd.append("--no-warnings")
        if self.extra_sumo_args:
            cmd.extend(self.extra_sumo_args.split())
        if self.use_gui or self.render_mode:
            cmd.extend(["--start", "--quit-on-end"])

        if USE_LIBSUMO:
            traci.start(cmd)
            self.sumo = traci
        else:
            traci.start(cmd, label=self._label)
            self.sumo = traci.getConnection(self._label)

    def _build_controllers(self, conn):
        """Tạo SignalController cho mỗi đèn tín hiệu."""
        if not isinstance(self.reward_fn, dict):
            reward_map = {s: self.reward_fn for s in self.signal_ids}
        else:
            reward_map = self.reward_fn

        self.controllers: Dict[str, SignalController] = {
            sid: SignalController(
                env=self,
                node_id=sid,
                step_interval=self.step_interval,
                amber_duration=self.amber_sec,
                min_green_sec=self.min_green,
                max_green_sec=self.max_green,
                enforce_cap=self.enforce_max_green,
                sim_start=0,
                reward_fn=reward_map[sid],
                reward_weights=self.reward_weights,
                sumo_conn=conn,
            )
            for sid in self.signal_ids
        }

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed, **kwargs)

        if self.episode_count > 0:
            self.close()
            self._save_metrics()
        self.episode_count += 1
        self.step_metrics = []

        if seed is not None:
            self.sumo_seed = seed

        self._launch_sumo()
        self._build_controllers(self.sumo)

        self.vehicle_registry = {}
        self._arrived = self._departed = self._teleported = 0

        if self.single_agent:
            obs = self._gather_states()[self.signal_ids[0]]
            return obs, self._build_info()
        # Multi-agent: tra ve (obs_dict, info)
        return self._gather_states(), self._build_info()

    @property
    def sim_step(self) -> float:
        """Thời gian mô phỏng hiện tại (giây)."""
        return self.sumo.simulation.getTime()

    def step(self, action: Union[dict, int]):
        """
        Thực hiện một bước: áp dụng action, tiến mô phỏng, trả về kết quả.

        Args:
            action: int (single_agent) hoặc dict {signal_id: duration}.
        """
        if self.fixed_signal or action is None or (
            isinstance(action, dict) and len(action) == 0
        ):
            for _ in range(self.step_interval):
                self._sumo_tick()
        else:
            self._dispatch_actions(action)
            self._advance_simulation()

        observations = self._gather_states()
        rewards = self._calc_rewards()
        dones = self._check_termination()
        truncated = dones["__all__"]
        info = self._build_info()

        if self.single_agent:
            sid = self.signal_ids[0]
            return observations[sid], rewards[sid], False, truncated, info
        return observations, rewards, dones, info

    # ------------------------------------------------------------------
    # Vòng lặp mô phỏng
    # ------------------------------------------------------------------

    def _advance_simulation(self):
        """Tiến mô phỏng cho đến khi ít nhất một controller cần quyết định."""
        ready = False
        while not ready:
            self._sumo_tick()
            for ctrl in self.controllers.values():
                ctrl.tick()
                if ctrl.time_to_act:
                    ready = True

    def _sumo_tick(self):
        """Chạy một bước simulationStep và cập nhật bộ đếm xe."""
        self.sumo.simulationStep()
        self._arrived += self.sumo.simulation.getArrivedNumber()
        self._departed += self.sumo.simulation.getDepartedNumber()
        self._teleported += self.sumo.simulation.getEndingTeleportNumber()

    def _dispatch_actions(self, actions):
        """Gửi action tới controller phù hợp."""
        if self.single_agent:
            sid = self.signal_ids[0]
            if self.controllers[sid].time_to_act:
                duration = int(actions) + self.min_green
                self.controllers[sid].apply_green_duration(duration)
        else:
            for sid, raw_action in actions.items():
                if self.controllers[sid].time_to_act:
                    duration = int(raw_action) + self.min_green
                    self.controllers[sid].apply_green_duration(duration)

    # ------------------------------------------------------------------
    # Tính toán obs / reward / done / info
    # ------------------------------------------------------------------

    def _gather_states(self) -> dict:
        """Thu thập observation từ tất cả controller cần hành động."""
        updates = {
            sid: self.controllers[sid].get_current_state()
            for sid in self.signal_ids
            if self.controllers[sid].time_to_act or self.fixed_signal
        }
        self.signal_states.update(updates)
        return {sid: self.signal_states[sid].copy() for sid in self.signal_states
                if self.controllers[sid].time_to_act or self.fixed_signal}

    def _calc_rewards(self) -> dict:
        """Tính phần thưởng từ tất cả controller cần hành động."""
        updates = {
            sid: self.controllers[sid].get_reward()
            for sid in self.signal_ids
            if self.controllers[sid].time_to_act or self.fixed_signal
        }
        self.signal_rewards.update(updates)
        return {sid: self.signal_rewards[sid] for sid in self.signal_rewards
                if self.controllers[sid].time_to_act or self.fixed_signal}

    def _check_termination(self) -> dict:
        """Kiểm tra điều kiện kết thúc episode."""
        done_flags = {sid: False for sid in self.signal_ids}
        done_flags["__all__"] = self.sim_step >= self.sim_max_time
        return done_flags

    def _build_info(self) -> dict:
        """Xây dựng dict thông tin bổ sung cho bước hiện tại."""
        info = {"sim_time": self.sim_step}
        info.update(self._system_metrics())
        info.update(self._per_signal_metrics())
        if getattr(self, "csv_output", None):
            flat_info = info.copy()
            flat_info.pop("lane_queue_map", None)  # Rất tốn RAM khi pandas to_csv
            self.step_metrics.append(flat_info)
        return info

    def _system_metrics(self) -> dict:
        """Các metric toàn hệ thống."""
        all_vehs = self.sumo.vehicle.getIDList()
        speeds = [self.sumo.vehicle.getSpeed(v) for v in all_vehs]
        waits = [self.sumo.vehicle.getWaitingTime(v) for v in all_vehs]
        backlog = len(self.sumo.simulation.getPendingVehicles())

        lane_queues = {
            lane: self.sumo.lane.getLastStepHaltingNumber(lane)
            for lane in self.sumo.lane.getIDList()
        }

        return {
            "sys_vehicles_running": len(all_vehs),
            "sys_vehicles_backlogged": backlog,
            "sys_vehicles_stopped": sum(1 for s in speeds if s < 0.1),
            "sys_vehicles_arrived": self._arrived,
            "sys_vehicles_departed": self._departed,
            "sys_vehicles_teleported": self._teleported,
            "sys_total_wait": sum(waits),
            "sys_mean_wait": float(np.mean(waits)) if waits else 0.0,
            "sys_mean_speed": float(np.mean(speeds)) if speeds else 0.0,
            "lane_queue_map": lane_queues,
        }

    def _per_signal_metrics(self) -> dict:
        """Metric riêng cho từng đèn tín hiệu."""
        info = {}
        for sid, ctrl in self.controllers.items():
            info[f"{sid}_total_queued"] = ctrl.get_total_queued()
            info[f"{sid}_mean_wait"] = sum(ctrl.get_lane_wait_times())
            info[f"{sid}_avg_speed"] = ctrl.get_avg_speed()
        info["all_signals_queued"] = sum(
            ctrl.get_total_queued() for ctrl in self.controllers.values()
        )
        return info

    # ------------------------------------------------------------------
    # Gym properties
    # ------------------------------------------------------------------

    @property
    def observation_space(self):
        if self.controllers:
            return self.controllers[self.signal_ids[0]].observation_space
        return self._obs_space

    @property
    def action_space(self):
        if self.controllers:
            return self.controllers[self.signal_ids[0]].action_space
        return self._action_space

    # ------------------------------------------------------------------
    # Tiện ích
    # ------------------------------------------------------------------

    def close(self):
        """Đóng kết nối SUMO."""
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
        self.close()

    def render(self):
        if self.render_mode == "human":
            return
        if self.render_mode == "rgb_array" and self._vdisplay:
            import numpy as np
            return np.array(self._vdisplay.grab())

    def _save_metrics(self):
        """Lưu metric ra CSV nếu được cấu hình."""
        if self.csv_output and self.step_metrics:
            df = pd.DataFrame(self.step_metrics)
            out = Path(self.csv_output)
            out.parent.mkdir(parents=True, exist_ok=True)
            filename = out.with_suffix("") / f"_ep{self.episode_count}.csv"
            df.to_csv(
                str(out.parent / f"{out.stem}_ep{self.episode_count}.csv"),
                index=False,
            )
