"""
intersection.py — SignalController

Quản lý logic điều khiển đèn tín hiệu tại một nút giao.
Giao tiếp với SUMO thông qua TraCI để đọc trạng thái giao thông
và cập nhật thời lượng pha đèn.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Callable, List, Union

import numpy as np
from gymnasium import spaces

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    raise ImportError("Chưa khai báo biến môi trường 'SUMO_HOME'")

import traci


class SignalController:
    """
    Đại diện cho một nút giao đèn tín hiệu trong môi trường SUMO.

    Chịu trách nhiệm:
    - Đọc dữ liệu hàng chờ, tốc độ, pha đèn từ TraCI
    - Tính phần thưởng (reward) dựa trên độ dài hàng chờ
    - Cập nhật thời gian đèn xanh theo lệnh của agent

    Quy ước pha SUMO: [green, yellow, green, yellow, ...]
    Chỉ số pha xanh: 0, 2, 4, 6 (mỗi pha chẵn là pha xanh)
    """

    # Khoảng cách tối thiểu giữa hai xe (theo chuẩn SUMO)
    MIN_GAP = 2.5

    def __init__(
        self,
        env,
        node_id: str,
        step_interval: int,
        amber_duration: int,
        min_green_sec: int,
        max_green_sec: int,
        enforce_cap: bool,
        sim_start: int,
        reward_fn: Union[str, Callable, List],
        reward_weights: List[float],
        sumo_conn,
    ):
        """
        Khởi tạo SignalController.

        Args:
            env: Môi trường TrafficControlEnv chứa controller này.
            node_id: ID của đèn tín hiệu trong SUMO.
            step_interval: Số giây giữa hai lần agent ra quyết định.
            amber_duration: Thời gian đèn vàng (giây).
            min_green_sec: Thời gian xanh tối thiểu (giây).
            max_green_sec: Thời gian xanh tối đa (giây).
            enforce_cap: Nếu True, tự động chuyển pha khi đạt max_green.
            sim_start: Thời điểm bắt đầu mô phỏng (giây).
            reward_fn: Hàm phần thưởng (chuỗi tên hoặc callable).
            reward_weights: Trọng số nếu dùng nhiều reward.
            sumo_conn: Kết nối TraCI.
        """
        self.node_id = node_id
        self.env = env
        self.step_interval = step_interval
        self.amber_duration = amber_duration
        self.min_green_sec = min_green_sec
        self.max_green_sec = max_green_sec
        self.enforce_cap = enforce_cap
        self.sumo = sumo_conn
        self.reward_fn = reward_fn
        self.reward_weights = reward_weights

        # Trạng thái nội tại
        self.active_phase = 0
        self.in_amber = False
        self.ticks_since_switch = 0
        self.next_decision_time = sim_start
        self.prev_wait_total = 0.0
        self.last_reward = None

        # Tham số cho hàm phần thưởng hàng chờ
        self.queue_norm_cap = 15              # số xe tối đa giả định mỗi hướng (Cologne)
        self.weight_reduction = 0.6          # α1 — thưởng khi hàng chờ giảm (paper: 0.6)
        self.weight_absolute = 0.4           # α2 — phạt theo hàng chờ tuyệt đối (paper: 0.4)
        self.prev_max_queues: dict = {}      # lưu hàng chờ bước trước

        # Map làn đường theo hướng tiếp cận
        self.approach_map = self._map_approach_lanes()

        # Xây dựng danh sách pha từ file SUMO
        self._load_phase_definitions()

        # Danh sách làn đường kiểm soát
        self.incoming_lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.node_id))
        )
        self.outgoing_lanes = [
            link[0][1]
            for link in self.sumo.trafficlight.getControlledLinks(self.node_id)
            if link
        ]
        self.outgoing_lanes = list(set(self.outgoing_lanes))
        self.lane_lengths = {
            lane: self.sumo.lane.getLength(lane)
            for lane in self.incoming_lanes + self.outgoing_lanes
        }

        # Reward function setup
        if isinstance(self.reward_fn, list):
            self.reward_dim = len(self.reward_fn)
            self._reward_funcs = [self._resolve_reward_fn(f) for f in self.reward_fn]
        else:
            self.reward_dim = 1
            self._reward_funcs = [self._resolve_reward_fn(self.reward_fn)]

        if self.reward_weights is not None:
            self.reward_dim = 1

        self.reward_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.reward_dim,), dtype=np.float32
        )

        # Observation function (được set bởi env sau khi khởi tạo)
        self.obs_fn = self.env.obs_class(self)

        self.observation_space = self.obs_fn.observation_space()
        # Action: chọn pha xanh tiếp theo (RESCO-style phase selection)
        self.action_space = spaces.Discrete(self.num_phases)

    # ------------------------------------------------------------------
    # Khởi tạo nội bộ
    # ------------------------------------------------------------------

    def _map_approach_lanes(self) -> dict:
        """
        Phát hiện làn đượng theo hướng tiếp cận dựa trên hình học SUMO.

        Thuật toán:
            1. Lấy tọa độ của nút giao từ SUMO.
            2. Với mỗi cạnh vào (incoming edge), lấy điểm đầu tiên
               của shape (xa junction nhất).
            3. Tính góc azimuth từ điểm đầu đến tâm junction.
            4. Phân loại: 315°-45° → N, 45°-135° → E, 135°-225° → S, 225°-315° → W.

        Hoạt động với bất kỳ mạng SUMO nào, không phụ thuộc tên edge.
        """
        lane_map: dict = {"N": [], "S": [], "E": [], "W": []}

        # Lấy tọa độ nút giao
        try:
            jx, jy = self.sumo.junction.getPosition(self.node_id)[:2]
        except traci.TraCIException:
            return lane_map

        # Gom làn đượng theo edge
        controlled = self.sumo.trafficlight.getControlledLanes(self.node_id)
        if not controlled:
            return lane_map

        edge_to_lanes: dict = {}
        for lane in controlled:
            try:
                if lane not in self.sumo.lane.getIDList():
                    continue
                edge = self.sumo.lane.getEdgeID(lane)
                edge_to_lanes.setdefault(edge, []).append(lane)
            except traci.TraCIException:
                continue

        # Xác định hướng theo góc hình học
        for edge, lanes in edge_to_lanes.items():
            try:
                # Dùng shape của làn đầu tiên trong edge (phổ biến hơn edge.getShape)
                first_lane = lanes[0]
                shape = self.sumo.lane.getShape(first_lane)
                if not shape or len(shape) < 1:
                    continue
                # Điểm đầu của edge (xa junction nhất)
                px, py = shape[0][0], shape[0][1]
                dx = jx - px
                dy = jy - py
                # Góc azimuth: 0° = Bắc, 90° = Đông, tính thuận chiều kim đồng hồ
                angle = math.degrees(math.atan2(dx, dy)) % 360

                if angle >= 315 or angle < 45:
                    direction = "N"
                elif 45 <= angle < 135:
                    direction = "E"
                elif 135 <= angle < 225:
                    direction = "S"
                else:
                    direction = "W"

                lane_map[direction].extend(sorted(lanes))
            except traci.TraCIException:
                continue

        return lane_map

    def _load_phase_definitions(self):
        """
        Đọc danh sách pha từ SUMO và xác định các pha xanh một cách động.

        Không hardcode chỉ số pha — tự phát hiện pha xanh từ chuỗi
        trạng thái (state): pha có ký tự 'G' nhưng không có 'y' là pha xanh.
        Hoạt động với mọi cấu hình pha trong bất kỳ mạng SUMO nào.
        """
        logic = self.sumo.trafficlight.getAllProgramLogics(self.node_id)[0]
        self.sumo.trafficlight.setProgramLogic(self.node_id, logic)
        self.all_phase_defs = logic.phases

        # Phát hiện pha xanh động dựa trên chuỗi trạng thái
        self.green_phase_idx = [
            i for i, ph in enumerate(self.all_phase_defs)
            if "G" in ph.state.upper() and "Y" not in ph.state.upper()
        ]

        # Fallback nếu không tìm thấy
        if not self.green_phase_idx:
            self.green_phase_idx = list(range(0, len(self.all_phase_defs), 2))

        self.num_phases = len(self.green_phase_idx)
        self.phase_durations = [
            self.all_phase_defs[i].duration for i in self.green_phase_idx
        ]

    # ------------------------------------------------------------------
    # Properties & update tick
    # ------------------------------------------------------------------

    @property
    def time_to_act(self) -> bool:
        """True khi đây là thời điểm agent cần ra quyết định."""
        return self.next_decision_time == self.env.sim_step

    def tick(self):
        """
        Cập nhật trạng thái nội tại mỗi bước mô phỏng.
        Gọi bởi TrafficControlEnv sau mỗi simulationStep().
        """
        try:
            current_phase = self.sumo.trafficlight.getPhase(self.node_id)
            if current_phase != getattr(self, "_last_phase_idx", current_phase):
                self.ticks_since_switch = 0
            else:
                self.ticks_since_switch += 1
            self._last_phase_idx = current_phase
        except traci.TraCIException as exc:
            if "connection closed" not in str(exc).lower():
                print(f"[SignalController] TraCI warning ({self.node_id}): {exc}")
            self._last_phase_idx = -1
            self.ticks_since_switch = 0

    # ------------------------------------------------------------------
    # Điều khiển pha đèn
    # ------------------------------------------------------------------

    def apply_green_duration(self, duration: int = 0):
        """
        Áp dụng thời gian xanh mới cho pha hiện tại.

        Agent truyền vào `duration` (số nguyên trong [min_green, max_green])
        và hàm này sẽ cập nhật thời lượng pha tương ứng trong SUMO.

        Args:
            duration: Thời gian xanh mong muốn (giây).
        """
        current_sumo_idx = self.sumo.trafficlight.getPhase(self.node_id)

        # Bỏ qua nếu đang ở pha vàng (lẻ)
        if current_sumo_idx % 2 != 0:
            self.next_decision_time = self.env.sim_step + self.step_interval
            return

        if current_sumo_idx >= len(self.all_phase_defs) or current_sumo_idx < 0:
            return

        target_state = self.all_phase_defs[current_sumo_idx].state
        clamped_dur = float(
            max(min(duration, self.max_green_sec), self.min_green_sec)
        )

        try:
            logics = self.sumo.trafficlight.getAllProgramLogics(self.node_id)
            if not logics:
                return
            current_logic = logics[0]

            updated_phases = []
            for ph in current_logic.phases:
                if ph.state == target_state:
                    updated_phases.append(
                        traci.trafficlight.Phase(clamped_dur, ph.state, ph.minDur, ph.maxDur)
                    )
                else:
                    updated_phases.append(ph)

            new_logic = traci.trafficlight.Logic(
                current_logic.programID,
                current_logic.type,
                current_logic.currentPhaseIndex,
                updated_phases,
            )
            self.sumo.trafficlight.setProgramLogic(self.node_id, new_logic)

        except traci.TraCIException as exc:
            print(f"[SignalController] Lỗi setProgramLogic ({self.node_id}): {exc}")
            return

        self.active_phase = current_sumo_idx
        self.next_decision_time = self.env.sim_step + self.step_interval

    def apply_phase_selection(self, action_idx: int):
        """
        RESCO-style phase selection: agent chọn pha xanh tiếp theo.

        Thay vì điều chỉnh thời lượng, agent chọn PHASE nào được xanh.
        Nếu action chọn pha khác hiện tại → setPhase trực tiếp (amber_sec=0)
        hoặc qua pha vàng (amber_sec>0).

        Args:
            action_idx: chỉ số trong [0, num_phases) — ánh xạ tới green_phase_idx.
        """
        target_sumo_idx = self.green_phase_idx[action_idx % self.num_phases]
        current_sumo_idx = self.sumo.trafficlight.getPhase(self.node_id)

        # Bỏ qua nếu đang trong pha vàng
        if current_sumo_idx not in self.green_phase_idx:
            self.next_decision_time = self.env.sim_step + self.step_interval
            return

        try:
            if current_sumo_idx != target_sumo_idx and self.amber_duration > 0:
                amber_idx = current_sumo_idx + 1
                if amber_idx >= len(self.all_phase_defs):
                    amber_idx = 1
                self.sumo.trafficlight.setPhase(self.node_id, amber_idx)
                self.sumo.trafficlight.setPhaseDuration(
                    self.node_id, float(self.amber_duration)
                )

            self.sumo.trafficlight.setPhase(self.node_id, target_sumo_idx)
            # +1 để SUMO không tự chuyển pha trong khoảng quyết định
            self.sumo.trafficlight.setPhaseDuration(
                self.node_id, float(self.step_interval + 1)
            )
        except traci.TraCIException as exc:
            print(f"[SignalController] Lỗi apply_phase_selection ({self.node_id}): {exc}")
            return

        self.active_phase = target_sumo_idx
        self.next_decision_time = self.env.sim_step + self.step_interval

    # ------------------------------------------------------------------
    # Observation & Reward
    # ------------------------------------------------------------------

    def get_current_state(self) -> np.ndarray:
        """Trả về vector trạng thái hiện tại từ obs_fn."""
        return self.obs_fn()

    def get_reward(self) -> Union[float, np.ndarray]:
        """Tính và trả về phần thưởng."""
        if self.reward_dim == 1:
            self.last_reward = self._reward_funcs[0](self)
        else:
            rewards = np.array(
                [fn(self) for fn in self._reward_funcs], dtype=np.float32
            )
            if self.reward_weights is not None:
                self.last_reward = float(np.dot(rewards, self.reward_weights))
            else:
                self.last_reward = rewards
        return self.last_reward

    # ------------------------------------------------------------------
    # Hàm phần thưởng
    # ------------------------------------------------------------------

    def _weighted_queue_penalty(self) -> float:
        """
        Phần thưởng tổng hợp dựa trên:
          R = α1 * R_reduction + α2 * R_absolute

        R_reduction: thưởng khi hàng chờ giảm so với bước trước.
        R_absolute:  phạt theo tổng hàng chờ hiện tại.

        Cả hai đều được chuẩn hóa theo (N_approaches × queue_norm_cap).
        """
        directions = ["N", "S", "E", "W"]
        current_maxq: dict = {}
        num_active = 0

        for d in directions:
            lanes = self.approach_map.get(d, [])
            if not lanes:
                current_maxq[d] = 0
                continue
            num_active += 1
            max_q = 0
            for lane in lanes:
                try:
                    q = self.sumo.lane.getLastStepHaltingNumber(lane)
                    if q > max_q:
                        max_q = q
                except traci.TraCIException:
                    pass
            current_maxq[d] = max_q

        if num_active == 0:
            self.prev_max_queues = current_maxq
            return 0.0

        total_current = sum(current_maxq[d] for d in directions)
        delta_total = sum(
            self.prev_max_queues.get(d, 0) - current_maxq[d] for d in directions
        )

        denom = num_active * self.queue_norm_cap or 1.0
        r_absolute = -total_current / denom
        r_reduction = delta_total / denom

        self.prev_max_queues = current_maxq
        return self.weight_absolute * r_absolute + self.weight_reduction * r_reduction

    def _diff_wait_reward(self) -> float:
        """Phần thưởng dựa trên sự thay đổi tổng thời gian chờ."""
        current_wait = sum(self.get_lane_wait_times()) / 100.0
        reward = self.prev_wait_total - current_wait
        self.prev_wait_total = current_wait
        return reward

    def _speed_reward(self) -> float:
        """Phần thưởng dựa trên tốc độ trung bình các xe."""
        return self.get_avg_speed()

    def _pressure_reward(self) -> float:
        """Pressure: -(in_halting - out_halting) chuẩn hóa, clip [-1, 0].

        Dùng xe dừng (halting) thay vì tổng xe; normalize theo
        n_incoming_lanes × queue_norm_cap để scale tương đương queue reward.
        """
        in_halting = sum(
            self.sumo.lane.getLastStepHaltingNumber(l) for l in self.incoming_lanes
        )
        out_halting = sum(
            self.sumo.lane.getLastStepHaltingNumber(l) for l in self.outgoing_lanes
        )
        denom = max(len(self.incoming_lanes) * self.queue_norm_cap, 1.0)
        return float(max(-1.0, min(0.0, -(in_halting - out_halting) / denom)))

    def _wait_clip_reward(self) -> float:
        """Wait-clip reward: clip(-mean_wait_per_vehicle / 60, -1, 0).

        Normalizes by vehicle count so the reward stays informative at any
        demand level — avoids the hard saturation at -5 that occurs when
        total_wait/100 is used with many vehicles present simultaneously.
        A vehicle waiting 0 s -> 0.0; waiting ≥ 60 s on average -> -1.0.
        """
        lanes = list(dict.fromkeys(
            self.sumo.trafficlight.getControlledLanes(self.node_id)
        ))
        total_wait = 0.0
        n_vehs = 0
        for lane in lanes:
            vids = self.sumo.lane.getLastStepVehicleIDs(lane)
            n_vehs += len(vids)
            for v in vids:
                total_wait += self.sumo.vehicle.getWaitingTime(v)
        if n_vehs == 0:
            return 0.0
        mean_wait = total_wait / n_vehs
        return float(max(-1.0, min(0.0, -mean_wait / 60.0)))

    # ------------------------------------------------------------------
    # Truy vấn dữ liệu TraCI
    # ------------------------------------------------------------------

    def get_lane_wait_times(self) -> List[float]:
        """Tổng thời gian chờ tích lũy theo từng làn đường."""
        wait_per_lane = []
        for lane in self.incoming_lanes:
            vehicles = self.sumo.lane.getLastStepVehicleIDs(lane)
            total_wait = 0.0
            for veh in vehicles:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicle_registry:
                    self.env.vehicle_registry[veh] = {veh_lane: acc}
                else:
                    self.env.vehicle_registry[veh][veh_lane] = acc - sum(
                        v for k, v in self.env.vehicle_registry[veh].items()
                        if k != veh_lane
                    )
                total_wait += self.env.vehicle_registry[veh][veh_lane]
            wait_per_lane.append(total_wait)
        return wait_per_lane

    def get_avg_speed(self) -> float:
        """Tốc độ trung bình chuẩn hóa của các xe trong nút giao."""
        vehs = self._list_nearby_vehicles()
        if not vehs:
            return 1.0
        return sum(
            self.sumo.vehicle.getSpeed(v) / self.sumo.vehicle.getAllowedSpeed(v)
            for v in vehs
        ) / len(vehs)

    def get_total_queued(self) -> int:
        """Tổng số xe đang dừng trong toàn bộ nút giao."""
        return sum(
            self.sumo.lane.getLastStepHaltingNumber(lane)
            for lane in self.incoming_lanes
        )

    def get_intersection_pressure(self) -> float:
        """Áp lực = tổng xe ra - tổng xe vào."""
        out_count = sum(
            self.sumo.lane.getLastStepVehicleNumber(l) for l in self.outgoing_lanes
        )
        in_count = sum(
            self.sumo.lane.getLastStepVehicleNumber(l) for l in self.incoming_lanes
        )
        return float(out_count - in_count)

    def get_lane_densities(self) -> List[float]:
        """Mật độ xe [0,1] trên từng làn đến."""
        result = []
        for lane in self.incoming_lanes:
            n_veh = self.sumo.lane.getLastStepVehicleNumber(lane)
            capacity = self.lane_lengths[lane] / (
                self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)
            )
            result.append(min(1.0, n_veh / capacity) if capacity > 0 else 0.0)
        return result

    def get_raw_queues(self) -> List[float]:
        """Số xe dừng thực tế (không chuẩn hóa) trên từng làn đến."""
        return [
            float(self.sumo.lane.getLastStepHaltingNumber(lane))
            for lane in self.incoming_lanes
        ]

    def _list_nearby_vehicles(self) -> List[str]:
        """Danh sách tất cả xe trong vùng kiểm soát."""
        vehs = []
        for lane in self.incoming_lanes:
            vehs.extend(self.sumo.lane.getLastStepVehicleIDs(lane))
        return vehs

    # ------------------------------------------------------------------
    # Resolve reward function
    # ------------------------------------------------------------------

    def _resolve_reward_fn(self, fn):
        """Chuyển đổi tên chuỗi sang callable hoặc giữ nguyên callable."""
        if callable(fn):
            return fn
        if isinstance(fn, str):
            mapping = SignalController.REWARD_REGISTRY
            if fn in mapping:
                return mapping[fn]
            raise NotImplementedError(f"Reward function '{fn}' chưa được đăng ký.")
        raise TypeError(f"Reward function phải là str hoặc callable, nhận được {type(fn)}.")

    REWARD_REGISTRY = {
        "queue": _weighted_queue_penalty,
        "diff-waiting-time": _diff_wait_reward,
        "average-speed": _speed_reward,
        "pressure": _pressure_reward,
        "wait-clip": _wait_clip_reward,
    }
