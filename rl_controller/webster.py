"""
webster.py — Dynamic Webster Traffic Signal Controller

Tinh toan thoi gian tin hieu toi uu theo cong thuc Webster (1958),
cap nhat moi 145 giay dua tren luu luong thuc te (moving average 15 phut).

Cong thuc Webster:
    Co = (1.5 * L + 5) / (1 - sum(Yi))

    Trong do:
        Co = Chu ky toi uu (giay)
        L  = Thoi gian mat mat moi chu ky = n_green_phases * lost_time_per_phase
        Yi = max(qij / Sj) — ty le luu luong / bao hoa cua nhom lan duong pha i
        qij = luu luong tren lane group j trong pha i (xe/s)
        Sj  = luu luong bao hoa = 1800 xe/h = 0.5 xe/s (tieu chuan)

Tham khao bai bao:
    "Webster's method is applied dynamically with traffic flow updates every
    145 seconds to ensure fair comparison with RL-based control. For real-time
    adaptation, Webster timings are recomputed using moving average flow estimates
    over the past 15 minutes."
"""

from __future__ import annotations

import collections
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SATURATION_FLOW_PER_LANE = 0.5     # xe/s (= 1800 xe/h / 3600s)
LOST_TIME_PER_PHASE      = 3.0     # giay mat mat moi pha (start-up + clearance)
MIN_CYCLE                = 30      # giay
MAX_CYCLE                = 120     # giay
MIN_GREEN                = 5       # giay xanh toi thieu moi pha
AMBER_DURATION           = 5       # giay vang
UPDATE_INTERVAL          = 145     # giay — cap nhat theo bai bao
FLOW_WINDOW_SEC          = 900     # 15 phut moving average (= 900 giay)


# ---------------------------------------------------------------------------
# WEbster Timing Calculator
# ---------------------------------------------------------------------------

def compute_cycle_length(y_values: List[float], n_phases: int) -> float:
    """
    Tinh chu ky toi uu bang cong thuc Webster.

    Args:
        y_values: Danh sach Yi (flow-to-saturation ratio) moi pha xanh.
        n_phases: So pha xanh.

    Returns:
        Chu ky toi uu (giay), da duoc clip vao [MIN_CYCLE, MAX_CYCLE].
    """
    L = n_phases * LOST_TIME_PER_PHASE
    sum_y = sum(y_values)

    # Tranh chia cho 0 hoac am (qua bao hoa)
    denom = 1.0 - sum_y
    if denom <= 0.05:
        # Giao lo qua tai — dung chu ky toi da
        return float(MAX_CYCLE)

    Co = (1.5 * L + 5.0) / denom
    return float(max(MIN_CYCLE, min(MAX_CYCLE, Co)))


def split_green_times(cycle: float, y_values: List[float], n_phases: int) -> List[float]:
    """
    Phan bo thoi gian xanh cho tung pha theo ty le Yi.

    Thoi gian hieu qua = cycle - tong thoi gian vang.
    Moi pha nhan phan theo ty le Yi / sum(Yi).

    Args:
        cycle:    Chu ky tin hieu (giay).
        y_values: Yi cua tung pha.
        n_phases: So pha xanh.

    Returns:
        Danh sach thoi gian xanh (giay) cho tung pha.
    """
    effective_time = cycle - n_phases * AMBER_DURATION
    sum_y = sum(y_values) or 1.0

    green_times = []
    for y in y_values:
        gt = (y / sum_y) * effective_time
        green_times.append(max(MIN_GREEN, round(gt)))

    return green_times


# ---------------------------------------------------------------------------
# DynamicWebsterController
# ---------------------------------------------------------------------------

class DynamicWebsterController:
    """
    Quan ly lich tin hieu Webster dong cho mot nut giao.

    Cap nhat lich dinh ky moi UPDATE_INTERVAL giay dua tren
    moving average luu luong 15 phut.

    Su dung:
        controller = DynamicWebsterController(ts_id, green_phase_idx, sumo_conn)
        # Goi trong vong lap mo phong:
        controller.step(sim_time)
        current_green_times = controller.green_times
    """

    def __init__(
        self,
        ts_id: str,
        green_phase_idx: List[int],
        sumo_conn,
        phase_to_lanes: Optional[Dict[int, List[str]]] = None,
    ):
        """
        Args:
            ts_id:           ID den tin hieu trong SUMO.
            green_phase_idx: Danh sach chi so pha xanh.
            sumo_conn:       Ket noi TraCI.
            phase_to_lanes:  {phase_idx: [lane_id, ...]} — lan duong phuc vu moi pha.
                             Neu None, tu dong phat hien tu TraCI.
        """
        self.ts_id = ts_id
        self.green_phase_idx = green_phase_idx
        self.n_phases = len(green_phase_idx)
        self.sumo = sumo_conn

        # Lap ban do lane -> pha
        if phase_to_lanes is None:
            self.phase_to_lanes = self._detect_phase_lanes()
        else:
            self.phase_to_lanes = phase_to_lanes

        # Moving average: {phase_idx: deque of (timestamp, count) tuples}
        self._flow_history: Dict[int, collections.deque] = {
            ph: collections.deque() for ph in green_phase_idx
        }

        # Thoi gian xanh hien tai
        n = self.n_phases
        initial_cycle = 90.0
        effective = initial_cycle - n * AMBER_DURATION
        initial_green = max(MIN_GREEN, round(effective / n))
        self.green_times = [initial_green] * n

        self._last_update_time = -UPDATE_INTERVAL  # bat dau update ngay lap tuc
        self._apply_timings(initial_cycle, self.green_times)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, sim_time: float):
        """
        Goi moi simulation step de cap nhat luu luong va lich tin hieu.

        Args:
            sim_time: Thoi gian hien tai cua mo phong (giay).
        """
        # Cap nhat luu luong tung pha
        self._record_flows(sim_time)

        # Kiem tra xem da den luc tinh lai hay chua
        if sim_time - self._last_update_time >= UPDATE_INTERVAL:
            self._recompute_timings(sim_time)
            self._last_update_time = sim_time

    @property
    def current_cycle_length(self) -> float:
        """Chu ky hien tai (giay)."""
        return sum(self.green_times) + self.n_phases * AMBER_DURATION

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _detect_phase_lanes(self) -> Dict[int, List[str]]:
        """Tu dong phat hien lane duoc phuc vu trong moi pha xanh."""
        logics = self.sumo.trafficlight.getAllProgramLogics(self.ts_id)
        if not logics:
            return {ph: [] for ph in self.green_phase_idx}

        all_links = self.sumo.trafficlight.getControlledLinks(self.ts_id)
        phases = logics[0].phases
        result = {}

        for ph_idx in self.green_phase_idx:
            if ph_idx >= len(phases):
                result[ph_idx] = []
                continue
            state = phases[ph_idx].state
            lanes = []
            for link in all_links:
                if not (isinstance(link, (list, tuple)) and len(link) >= 2):
                    continue
                inner = link[0]
                if not (isinstance(inner, (list, tuple)) and len(inner) >= 1):
                    continue
                in_lane = inner[0]
                link_pos = link[1]
                if link_pos < len(state) and state[link_pos].upper() == "G":
                    lanes.append(in_lane)
            result[ph_idx] = sorted(set(lanes))

        return result

    def _record_flows(self, sim_time: float):
        """Ghi nhan so xe dang qua moi nhom lane trong buoc nay."""
        for ph_idx in self.green_phase_idx:
            lanes = self.phase_to_lanes.get(ph_idx, [])
            count = 0
            for lane in lanes:
                try:
                    count += self.sumo.lane.getLastStepVehicleNumber(lane)
                except Exception:
                    pass
            hist = self._flow_history[ph_idx]
            hist.append((sim_time, count))
            # Xoa cac ban ghi qua cu (qua FLOW_WINDOW_SEC)
            cutoff = sim_time - FLOW_WINDOW_SEC
            while hist and hist[0][0] < cutoff:
                hist.popleft()

    def _get_avg_flow(self, ph_idx: int) -> float:
        """
        Tra ve luu luong trung binh (xe/s) trong cua so 15 phut.

        Args:
            ph_idx: Chi so pha xanh.

        Returns:
            Luu luong trung binh (xe/s).
        """
        hist = self._flow_history[ph_idx]
        if not hist:
            return 0.0
        total_vehicles = sum(cnt for _, cnt in hist)
        window_sec = min(FLOW_WINDOW_SEC, max(1.0, len(hist)))
        return total_vehicles / window_sec

    def _recompute_timings(self, sim_time: float):
        """
        Tinh lai chu ky va phan bo thoi gian xanh theo cong thuc Webster.
        """
        y_values = []
        for ph_idx in self.green_phase_idx:
            q = self._get_avg_flow(ph_idx)      # xe/s
            lanes = self.phase_to_lanes.get(ph_idx, [])
            n_lanes = max(1, len(lanes))
            # Yi = luu luong trung binh moi lane / luu luong bao hoa
            yi = (q / n_lanes) / SATURATION_FLOW_PER_LANE
            y_values.append(min(yi, 0.9))       # cap tai 90% de tranh qua bao hoa

        cycle = compute_cycle_length(y_values, self.n_phases)
        self.green_times = split_green_times(cycle, y_values, self.n_phases)

        self._apply_timings(cycle, self.green_times)

    def _apply_timings(self, cycle: float, green_times: List[float]):
        """
        Cap nhat lich tin hieu trong SUMO theo thoi gian xanh moi.

        Xay dung lai program logic voi thoi gian moi.
        """
        try:
            logics = self.sumo.trafficlight.getAllProgramLogics(self.ts_id)
            if not logics:
                return
            current_logic = logics[0]
            all_phases = list(current_logic.phases)

            import traci
            updated_phases = []
            green_idx_ptr = 0

            for i, ph in enumerate(all_phases):
                state = ph.state.upper()
                is_green = ("G" in state and "Y" not in state)

                if is_green and green_idx_ptr < len(green_times):
                    new_dur = float(green_times[green_idx_ptr])
                    green_idx_ptr += 1
                    updated_phases.append(
                        traci.trafficlight.Phase(new_dur, ph.state, ph.minDur, ph.maxDur)
                    )
                else:
                    updated_phases.append(ph)

            new_logic = traci.trafficlight.Logic(
                current_logic.programID,
                current_logic.type,
                current_logic.currentPhaseIndex,
                updated_phases,
            )
            self.sumo.trafficlight.setProgramLogic(self.ts_id, new_logic)

        except Exception as exc:
            # Khong tu crash — tiep tuc voi lich cu
            print(f"[Webster] Canh bao cap nhat lich {self.ts_id}: {exc}")
