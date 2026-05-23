"""
state_builder.py — Lớp xây dựng trạng thái quan sát

BaselineObservation (7D):
   s = [g1, g2, g3, g4, phase_index, remaining_time, total_queue]
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from gymnasium import spaces

try:
    import traci
except ImportError:
    traci = None  # type: ignore


logger = logging.getLogger(__name__)


class BaselineObservation:
    """
    Không gian trạng thái 7 chiều theo bài báo RLTSCQ (Baseline Representation).

    s = [g1, g2, g3, g4, phase_index, remaining_time, total_queue]

        g1..g4         — thời gian xanh lập trình cho 4 pha (chuẩn hóa / MAX_G)
        phase_index    — chỉ số pha xanh hiện tại (chuẩn hóa / 3)
        remaining_time — thời gian còn lại trong pha hiện tại (chuẩn hóa / MAX_G)
        total_queue    — tổng xe dừng toàn nút (chuẩn hóa / MAX_Q)
    """

    NUM_PHASES = 4
    MAX_G = 120.0
    MAX_Q = 50.0

    def __init__(self, controller):
        self.ctrl = controller
        self._green_states: List[str] = []
        self._green_durations: List[float] = []
        self._init_phase_info()

    def _is_green(self, state_str: str) -> bool:
        lo = state_str.lower()
        return "g" in lo and "y" not in lo

    def _init_phase_info(self):
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
        try:
            if not self._green_states:
                self._init_phase_info()

            # g1..g4: thời gian xanh lập trình (chuẩn hóa, pad 0 nếu < 4 pha)
            g_norm = [min(d / self.MAX_G, 1.0) for d in self._green_durations]
            g_norm = (g_norm + [0.0] * self.NUM_PHASES)[: self.NUM_PHASES]

            # phase_index: scalar chuẩn hóa
            cur_state = self.ctrl.sumo.trafficlight.getRedYellowGreenState(self.ctrl.node_id)
            phase_idx = 0
            if self._is_green(cur_state) and cur_state in self._green_states:
                phase_idx = self._green_states.index(cur_state)
            phase_norm = phase_idx / max(self.NUM_PHASES - 1, 1)

            # remaining_time
            next_switch = self.ctrl.sumo.trafficlight.getNextSwitch(self.ctrl.node_id)
            sim_t = self.ctrl.env.sim_step
            remaining = min(max(0.0, float(next_switch) - float(sim_t)) / self.MAX_G, 1.0)

            # total_queue: tổng xe dừng trên tất cả làn vào
            lanes = list(dict.fromkeys(
                self.ctrl.sumo.trafficlight.getControlledLanes(self.ctrl.node_id)
            ))
            total_q = sum(
                self.ctrl.sumo.lane.getLastStepHaltingNumber(l) for l in lanes
            )
            q_norm = min(total_q / self.MAX_Q, 1.0)

            return np.array(g_norm + [phase_norm, remaining, q_norm], dtype=np.float32)

        except Exception as exc:
            logger.error("BaselineObservation: Lỗi tính trạng thái (%s): %s",
                         self.ctrl.node_id, exc)
            return np.zeros(self.NUM_PHASES + 3, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        """Không gian quan sát cố định 7 chiều, tất cả trong [0, 1]."""
        return spaces.Box(
            low=np.zeros(self.NUM_PHASES + 3, dtype=np.float32),
            high=np.ones(self.NUM_PHASES + 3, dtype=np.float32),
            dtype=np.float32,
        )

