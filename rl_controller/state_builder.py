"""
state_builder.py — Các lớp xây dựng trạng thái quan sát

Gồm hai lớp observation function:

1. BaselineObservation (đơn giản):
   Trạng thái gốc không chuẩn hóa, dùng làm baseline so sánh.
   s = [phase_onehot, phase_durations, remaining_time, raw_queue_per_lane]
   Chiều động (phụ thuộc số pha và số làn).

2. IntersectionStateExtractor (19D đầy đủ):
   Vector trạng thái chuẩn hóa 19 chiều, đầu vào cho Autoencoder.
   s = [cycle_ratio, phase_onehot(4), phase_ratio, cycle_count,
        Q_N, Q_S, Q_E, Q_W,
        ΔQ_N, ΔQ_S, ΔQ_E, ΔQ_W,
        g1_ratio, g2_ratio, g3_ratio, g4_ratio]

Ký hiệu (IntersectionStateExtractor):
    cycle_ratio  — tỉ lệ thời gian đã qua trong chu kỳ hiện tại
    phase_onehot — one-hot encoding pha xanh hiện tại (4 pha)
    phase_ratio  — tỉ lệ thời gian đã qua trong pha hiện tại
    cycle_count  — số chu kỳ đã hoàn thành (chuẩn hóa)
    Q_i          — hàng chờ tối đa tại hướng i (chuẩn hóa)
    ΔQ_i         — biến thiên hàng chờ tại hướng i (chuẩn hóa)
    gj_ratio     — tỉ lệ thời gian xanh được lập trình cho pha j
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from gymnasium import spaces

try:
    import traci
except ImportError:
    traci = None  # type: ignore


logger = logging.getLogger(__name__)


class IntersectionStateExtractor:
    """
    Trích xuất đặc trưng trạng thái 19D tại một nút giao đèn tín hiệu.

    Đây là lớp observation function dùng trực tiếp với TrafficControlEnv.
    """

    NUM_PROGRAMMABLE_PHASES = 4
    DIRECTIONS = ["N", "S", "E", "W"]

    def __init__(
        self,
        controller,
        max_queue_ref: float = 150.0,
        max_green_ref: float = 120.0,
    ):
        """
        Args:
            controller: Đối tượng SignalController gắn với nút giao này.
            max_queue_ref: Hàng chờ tham chiếu tối đa để chuẩn hóa.
            max_green_ref: Thời gian xanh tham chiếu tối đa (giây).
        """
        self.ctrl = controller
        self.MAX_Q = float(max_queue_ref)
        self.MAX_G = float(max_green_ref)
        self.MAX_CYCLE = float(max_green_ref * self.NUM_PROGRAMMABLE_PHASES)
        self.MAX_CYCLE_COUNT = 10.0

        # Trạng thái nội tại theo dõi chu kỳ
        self._prev_queues: Dict[str, float] = {d: 0.0 for d in self.DIRECTIONS}
        self._cycle_count: float = 0.0
        self._time_in_phase: float = 0.0
        self._total_cycle_time: float = 0.0
        self._prev_phase_idx: int = -1

        # Thông tin pha đèn (đọc từ SUMO một lần)
        self._green_phase_states: List[str] = []
        self._green_durations: List[float] = []
        self._num_green: int = 0

        self._init_phase_info()

    def _init_phase_info(self):
        """Đọc danh sách pha xanh và thời lượng từ SUMO."""
        try:
            logics = self.ctrl.sumo.trafficlight.getAllProgramLogics(self.ctrl.node_id)
            if not logics:
                return
            for ph in logics[0].phases:
                state_lo = ph.state.lower()
                if "g" in state_lo and "y" not in state_lo:
                    self._green_phase_states.append(ph.state)
                    self._green_durations.append(float(ph.duration))
            self._num_green = len(self._green_phase_states)
        except Exception as exc:
            logger.warning("Không thể đọc pha từ SUMO (%s): %s", self.ctrl.node_id, exc)

    def _current_phase_index(self) -> int:
        """Trả về chỉ số pha xanh hiện tại, hoặc -1 nếu đang pha vàng/đỏ."""
        try:
            state = self.ctrl.sumo.trafficlight.getRedYellowGreenState(self.ctrl.node_id)
            lo = state.lower()
            if "y" not in lo and "g" in lo and state in self._green_phase_states:
                return self._green_phase_states.index(state)
        except Exception:
            pass
        return -1

    def _approach_queues(self) -> Dict[str, float]:
        """Hàng chờ lớn nhất (số xe dừng) tại mỗi hướng tiếp cận."""
        queues = {d: 0.0 for d in self.DIRECTIONS}
        for direction, lanes in self.ctrl.approach_map.items():
            if not lanes:
                continue
            try:
                queues[direction] = float(
                    max(self.ctrl.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)
                )
            except Exception:
                pass
        return queues

    def __call__(self) -> np.ndarray:
        """
        Tính và trả về vector trạng thái 19 chiều.

        Trả về mảng numpy float32 shape (19,).
        """
        try:
            phase_idx = self._current_phase_index()
            step_dt = 1.0  # mỗi lần gọi tương ứng 1 giây mô phỏng

            # Cập nhật bộ đếm chu kỳ / pha
            if phase_idx != -1 and phase_idx != self._prev_phase_idx:
                completed_last = (
                    self._prev_phase_idx == self._num_green - 1
                    and phase_idx == 0
                )
                if completed_last:
                    self._cycle_count += 1.0
                    self._total_cycle_time = 0.0
                self._time_in_phase = 0.0
                self._prev_phase_idx = phase_idx

            self._time_in_phase += step_dt
            self._total_cycle_time += step_dt

            # Hàng chờ hiện tại
            cur_q = self._approach_queues()
            q_vec = [cur_q[d] for d in self.DIRECTIONS]
            delta_q = [cur_q[d] - self._prev_queues.get(d, 0.0) for d in self.DIRECTIONS]
            self._prev_queues = cur_q

            # Chuẩn hóa
            cycle_ratio = min(self._total_cycle_time / self.MAX_CYCLE, 1.0)
            phase_ratio = min(self._time_in_phase / self.MAX_G, 1.0)
            cycle_count_norm = min(self._cycle_count / self.MAX_CYCLE_COUNT, 1.0)

            phase_onehot = [0.0] * self.NUM_PROGRAMMABLE_PHASES
            if 0 <= phase_idx < self.NUM_PROGRAMMABLE_PHASES:
                phase_onehot[phase_idx] = 1.0

            q_norm = [min(q / self.MAX_Q, 1.0) for q in q_vec]
            delta_norm = [float(np.clip(dq / self.MAX_Q, -1.0, 1.0)) for dq in delta_q]

            g_norm = [min(g / self.MAX_G, 1.0) for g in self._green_durations]
            g_norm += [0.0] * (self.NUM_PROGRAMMABLE_PHASES - len(g_norm))
            g_norm = g_norm[: self.NUM_PROGRAMMABLE_PHASES]

            vec = (
                [cycle_ratio]
                + phase_onehot
                + [phase_ratio, cycle_count_norm]
                + q_norm
                + delta_norm
                + g_norm
            )
            return np.array(vec, dtype=np.float32)

        except Exception as exc:
            logger.error("Lỗi khi tính trạng thái (%s): %s", self.ctrl.node_id, exc)
            return np.zeros(self.observation_space().shape, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        """
        Không gian quan sát Gym.

        Chiều: 1 + 4 + 1 + 1 + 4 + 4 + 4 = 19
        """
        n = len(self.DIRECTIONS)
        dims = 1 + self.NUM_PROGRAMMABLE_PHASES + 1 + 1 + n * 2 + self.NUM_PROGRAMMABLE_PHASES

        low = np.zeros(dims, dtype=np.float32)
        # ΔQ có thể âm: set lower bound = -1
        dq_start = 1 + self.NUM_PROGRAMMABLE_PHASES + 2 + n
        low[dq_start: dq_start + n] = -1.0

        high = np.ones(dims, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)


class BaselineObservation:
    """
    Observation function đơn giản — dùng làm baseline so sánh.

    Trạng thái gồm:
        [phase_onehot(N), green_durations(N), remaining_time(1), raw_queue(M)]

    Trong đó:
        N = số pha xanh (đọc động từ SUMO)
        M = số làn đường kiểm soát

    Không chuẩn hóa, không tổng hợp theo hướng — đây là dạng quan sát
    trực tiếp nhất từ dữ liệu TraCI, phù hợp để làm điểm so sánh.
    """

    def __init__(self, controller):
        self.ctrl = controller
        self._green_states: List[str] = []
        self._green_durations: List[float] = []
        self._init_phase_info()

    def _is_green(self, state_str: str) -> bool:
        """Kiểm tra chuỗi trạng thái có phải pha xanh không."""
        lo = state_str.lower()
        return "g" in lo and "y" not in lo

    def _init_phase_info(self):
        """Đọc danh sách pha xanh từ SUMO."""
        try:
            logics = self.ctrl.sumo.trafficlight.getAllProgramLogics(self.ctrl.node_id)
            if not logics:
                return
            for ph in logics[0].phases:
                if self._is_green(ph.state):
                    self._green_states.append(ph.state)
                    self._green_durations.append(float(ph.duration))
        except Exception as exc:
            logger.warning("BaselineObservation: Không đọc được pha (%s): %s",
                           self.ctrl.node_id, exc)

    def __call__(self) -> np.ndarray:
        """
        Trả về vector trạng thái thô.

        Nếu có lỗi TraCI, trả về vector 0 kích thước đúng.
        """
        n_phases = max(len(self._green_states), 2)  # fallback = 2
        n_lanes = len(self.ctrl.incoming_lanes) if hasattr(self.ctrl, "incoming_lanes") else 0

        try:
            # Cập nhật pha xanh hiện tại (re-read nếu chưa có)
            if not self._green_states:
                self._init_phase_info()
                n_phases = max(len(self._green_states), 2)

            # One-hot pha hiện tại
            cur_state = self.ctrl.sumo.trafficlight.getRedYellowGreenState(self.ctrl.node_id)
            phase_onehot = [0.0] * n_phases
            if self._is_green(cur_state) and cur_state in self._green_states:
                phase_onehot[self._green_states.index(cur_state)] = 1.0

            # Thời lượng pha xanh được lập trình
            durations = list(self._green_durations) or [0.0] * n_phases

            # Thời gian còn lại trong pha hiện tại
            next_switch = self.ctrl.sumo.trafficlight.getNextSwitch(self.ctrl.node_id)
            sim_t = self.ctrl.env.sim_step
            remaining = max(0.0, float(next_switch) - float(sim_t))

            # Hàng chờ thô trên từng làn (chưa chuẩn hóa)
            raw_queues = self.ctrl.get_raw_queues()

            vec = phase_onehot + durations + [remaining] + raw_queues
            return np.array(vec, dtype=np.float32)

        except Exception as exc:
            logger.error("BaselineObservation: Lỗi tính trạng thái (%s): %s",
                         self.ctrl.node_id, exc)
            obs_len = n_phases + n_phases + 1 + n_lanes
            return np.zeros(obs_len, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        """
        Không gian quan sát động.

        Chiều = N_phases * 2 + 1 (remaining) + N_lanes
        """
        n_phases = max(len(self._green_states), 2)
        n_lanes = len(self.ctrl.incoming_lanes) if hasattr(self.ctrl, "incoming_lanes") else 0
        obs_len = n_phases + n_phases + 1 + n_lanes

        low = np.zeros(obs_len, dtype=np.float32)
        high = np.full(obs_len, np.inf, dtype=np.float32)
        # Phần one-hot giới hạn [0, 1]
        high[:n_phases] = 1.0

        return spaces.Box(low=low, high=high, dtype=np.float32)

