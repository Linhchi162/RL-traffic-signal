import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import traci
from sumolib import checkBinary

from build_spatiograph import build_graph


@dataclass(frozen=True)
class EnvConfig:
    sumo_cfg: Path
    gui: bool = False
    decision_interval_s: int = 10
    yellow_s: int = 3
    seed: int = 0


class STMARLEnv:
    """A minimal multi-agent SUMO environment for STMARL-style experiments.

    Implements a POMDP-like loop:
    - observation(): reads edge-level traffic metrics + per-TLS phase
    - reward(): negative sum of queues (halting vehicles) on incoming edges per TLS
    - step(actions): optionally switch phases (binary), enforce yellow, then advance time

    Notes:
    - This version uses edge-level metrics (as suggested):
      q_l = traci.edge.getLastStepHaltingNumber(edge_id)
      n_l = traci.edge.getLastStepVehicleNumber(edge_id)
      speed_l = traci.edge.getLastStepMeanSpeed(edge_id)
      phaseID_i = traci.trafficlight.getPhase(tls_id)
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        random.seed(cfg.seed)

        self._started = False
        self._graph: dict[str, Any] = {}
        self.control_nodes: list[str] = []
        self.edges_data: list[dict[str, Any]] = []
        self.control_edges: list[dict[str, Any]] = []

        # Pre-compute incoming edges per control node for rewards.
        self._incoming_edges_by_tls: dict[str, list[str]] = {}

        # Control-graph topology among signalized junctions (directed).
        self._control_node_index: dict[str, int] = {}
        self._control_edge_index: list[tuple[int, int]] = []

        # Keep phase counts for each TLS for safe modulo.
        self._phase_counts: dict[str, int] = {}

        self._load_graph()
        self.start()

    def _load_graph(self) -> None:
        net_path = self._infer_net_from_cfg(self.cfg.sumo_cfg)
        self._graph = build_graph(net_path)
        self.control_nodes = list(self._graph.get("control_nodes", []))
        self.edges_data = list(self._graph.get("edges_data", []))
        self.control_edges = list(self._graph.get("edges_between_control_nodes", []))

        self._control_node_index = {nid: i for i, nid in enumerate(self.control_nodes)}
        self._control_edge_index = []
        for e in self.control_edges:
            src = e.get("src")
            dst = e.get("dst")
            if src in self._control_node_index and dst in self._control_node_index:
                self._control_edge_index.append((self._control_node_index[src], self._control_node_index[dst]))

        incoming: dict[str, list[str]] = {tls: [] for tls in self.control_nodes}
        for e in self.edges_data:
            rec = e["rec_k"]
            if rec in incoming:
                incoming[rec].append(e["edge_id"])
        self._incoming_edges_by_tls = incoming

    def get_control_edge_index(self) -> list[list[int]]:
        """Return PyG-style edge_index (2 x E) for the TLS-only control graph."""
        if not self._control_edge_index:
            return [[], []]
        src = [s for s, _ in self._control_edge_index]
        dst = [d for _, d in self._control_edge_index]
        return [src, dst]

    def node_features_from_observation(self, obs: dict[str, Any]) -> list[list[float]]:
        """Aggregate edge-level obs into per-TLS node features.

        Feature vector per TLS i:
          [sum_q_in, sum_n_in, mean_speed_in, phaseID]
        """
        edge_obs: dict[str, dict[str, float]] = obs.get("edge", {})
        phases: dict[str, int] = obs.get("phaseID", {})

        features: list[list[float]] = []
        for tls_id in self.control_nodes:
            incoming_edges = self._incoming_edges_by_tls.get(tls_id, [])
            sum_q = 0.0
            sum_n = 0.0
            speed_weighted = 0.0
            for edge_id in incoming_edges:
                m = edge_obs.get(edge_id)
                if not m:
                    continue
                q = float(m.get("q", 0.0))
                n = float(m.get("n", 0.0))
                sp = float(m.get("speed", 0.0))
                sum_q += q
                sum_n += n
                speed_weighted += sp * max(n, 1.0)

            mean_speed = speed_weighted / max(sum_n, 1.0)
            phase = float(phases.get(tls_id, -1))
            features.append([sum_q, sum_n, mean_speed, phase])

        return features

    @staticmethod
    def _infer_net_from_cfg(cfg_path: Path) -> Path:
        # Our generator uses <net-file value="..."/> in run.sumo.cfg
        text = cfg_path.read_text(encoding="utf-8", errors="ignore")
        marker = "<net-file value=\""
        idx = text.find(marker)
        if idx == -1:
            raise SystemExit(f"Could not find <net-file ...> in cfg: {cfg_path}")
        start = idx + len(marker)
        end = text.find('"', start)
        net_rel = text[start:end]
        return cfg_path.parent / net_rel

    def start(self) -> None:
        if self._started:
            return

        sumo_binary = checkBinary("sumo-gui" if self.cfg.gui else "sumo")
        cmd = [
            sumo_binary,
            "-c",
            str(self.cfg.sumo_cfg),
            "--start",
            "--quit-on-end",
            "--no-step-log",
        ]
        traci.start(cmd)
        self._started = True

        # Populate phase counts after start (TraCI knows current programs).
        for tls_id in self.control_nodes:
            try:
                logics = traci.trafficlight.getAllProgramLogics(tls_id)
                if logics and logics[0].phases:
                    self._phase_counts[tls_id] = len(logics[0].phases)
                else:
                    self._phase_counts[tls_id] = 0
            except Exception:
                self._phase_counts[tls_id] = 0

    def close(self) -> None:
        if self._started:
            traci.close()
            self._started = False

    def observation(self) -> dict[str, Any]:
        # Edge-level metrics
        edge_metrics: dict[str, dict[str, float]] = {}
        for e in self.edges_data:
            edge_id = e["edge_id"]
            try:
                q = float(traci.edge.getLastStepHaltingNumber(edge_id))
                n = float(traci.edge.getLastStepVehicleNumber(edge_id))
                speed = float(traci.edge.getLastStepMeanSpeed(edge_id))
            except Exception:
                # Edge might not exist in the running sim for some reason; keep zeros.
                q, n, speed = 0.0, 0.0, 0.0
            edge_metrics[edge_id] = {"q": q, "n": n, "speed": speed}

        # Per-TLS phase
        phases: dict[str, int] = {}
        for tls_id in self.control_nodes:
            try:
                phases[tls_id] = int(traci.trafficlight.getPhase(tls_id))
            except Exception:
                phases[tls_id] = -1

        return {
            "t": float(traci.simulation.getTime()),
            "edge": edge_metrics,
            "phaseID": phases,
        }

    def reward(self) -> dict[str, float]:
        rewards: dict[str, float] = {}
        for tls_id in self.control_nodes:
            q_sum = 0.0
            for edge_id in self._incoming_edges_by_tls.get(tls_id, []):
                try:
                    q_sum += float(traci.edge.getLastStepHaltingNumber(edge_id))
                except Exception:
                    q_sum += 0.0
            rewards[tls_id] = -q_sum
        return rewards

    @staticmethod
    def _to_yellow_state(state: str) -> str:
        # Replace any green with yellow, keep red as-is.
        # This is a generic approximation that matches the "insert yellow" requirement.
        return "".join("y" if c in ("g", "G") else c for c in state)

    def _get_phase_states(self, tls_id: str) -> list[str]:
        try:
            logics = traci.trafficlight.getAllProgramLogics(tls_id)
            if not logics:
                return []
            phases = logics[0].phases
            return [p.state for p in phases]
        except Exception:
            return []

    def _switch_phase_with_yellow(self, tls_id: str) -> None:
        phase_count = self._phase_counts.get(tls_id, 0)
        if phase_count <= 1:
            return

        current_phase = int(traci.trafficlight.getPhase(tls_id))
        target_phase = (current_phase + 1) % phase_count

        # Try to find an explicit "yellow" phase in the existing program.
        phase_states = self._get_phase_states(tls_id)
        yellow_state = None
        try:
            yellow_state = self._to_yellow_state(phase_states[current_phase]) if phase_states else None
        except Exception:
            yellow_state = None

        yellow_phase_index: int | None = None
        if yellow_state and phase_states:
            try:
                yellow_phase_index = phase_states.index(yellow_state)
            except ValueError:
                yellow_phase_index = None

        # 1) go to yellow (either explicit yellow phase, or immediate next phase)
        intermediate_phase = yellow_phase_index if yellow_phase_index is not None else target_phase
        traci.trafficlight.setPhase(tls_id, intermediate_phase)
        traci.trafficlight.setPhaseDuration(tls_id, float(self.cfg.yellow_s))
        for _ in range(int(self.cfg.yellow_s)):
            traci.simulationStep()

        # 2) then go to target green phase (if intermediate already equals target, advance one more)
        final_phase = target_phase
        if intermediate_phase == target_phase:
            final_phase = (target_phase + 1) % phase_count
        traci.trafficlight.setPhase(tls_id, final_phase)

    def step(self, actions: dict[str, int] | int) -> tuple[dict[str, Any], dict[str, float], bool, dict[str, Any]]:
        """Advance the simulation by one decision.

        actions:
          - dict: tls_id -> 0/1
          - int: apply to all tls

        Returns: obs, rewards, done, info
        """
        if isinstance(actions, int):
            act_map = {tls: int(actions) for tls in self.control_nodes}
        else:
            act_map = {tls: int(actions.get(tls, 0)) for tls in self.control_nodes}

        # Apply actions
        for tls_id, act in act_map.items():
            if act == 1:
                self._switch_phase_with_yellow(tls_id)

        # Hold action for at least decision_interval_s seconds.
        for _ in range(int(self.cfg.decision_interval_s)):
            traci.simulationStep()

        obs = self.observation()
        rew = self.reward()
        done = traci.simulation.getMinExpectedNumber() == 0
        info = {"time": float(traci.simulation.getTime())}
        return obs, rew, done, info


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STMARL SUMO TraCI environment smoke test.")
    parser.add_argument(
        "--cfg",
        default="scenarios/grid_3x3_lanes3_dynamic_dense/run.sumo.cfg",
        help="Path to SUMO run.sumo.cfg",
    )
    parser.add_argument("--gui", action="store_true", help="Run with SUMO-GUI")
    parser.add_argument("--dec-interval", type=int, default=10, help="Decision interval (s)")
    parser.add_argument("--yellow", type=int, default=3, help="Yellow duration before switching (s)")
    parser.add_argument("--decisions", type=int, default=30, help="How many decisions to run")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


def _resolve_cfg_path(cfg_arg: str) -> Path:
    cfg_path = Path(cfg_arg)
    if cfg_path.exists():
        return cfg_path

    # If user passed only a filename (common: run.sumo.cfg), try to locate it under scenarios/.
    if cfg_path.parent == Path("."):
        repo_root = Path(__file__).resolve().parent
        scenarios_dir = repo_root / "scenarios"
        if scenarios_dir.exists():
            candidates = list(scenarios_dir.rglob(cfg_path.name))
            candidates = [p for p in candidates if p.is_file()]
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                rel = [str(p.relative_to(repo_root)) for p in candidates]
                raise SystemExit(
                    f"Config not found in current directory: {cfg_path.name}\n"
                    "Multiple matches were found under scenarios/:\n- "
                    + "\n- ".join(rel)
                    + "\n\nPlease re-run with an explicit path, e.g.\n"
                    + f"python env.py --cfg {rel[0]} --gui --decisions 999\n"
                )

    raise SystemExit(
        f"SUMO config not found: {cfg_arg}\n"
        "Tip: pass a full path like:\n"
        "  python env.py --cfg scenarios/grid_3x3_lanes3_dynamic_dense/run.sumo.cfg --gui\n"
    )


def main() -> int:
    args = _parse_args()

    cfg_path = _resolve_cfg_path(args.cfg)
    env = STMARLEnv(
        EnvConfig(
            sumo_cfg=cfg_path,
            gui=args.gui,
            decision_interval_s=args.dec_interval,
            yellow_s=args.yellow,
            seed=args.seed,
        )
    )

    try:
        print("TLS agents:", len(env.control_nodes), env.control_nodes[:5])
        for k in range(args.decisions):
            # Random hard-coded actions (0/1)
            actions = {tls: random.randint(0, 1) for tls in env.control_nodes}
            obs, rew, done, info = env.step(actions)

            # Print a compact snapshot
            t = obs["t"]
            # show first tls
            tls0 = env.control_nodes[0] if env.control_nodes else "<none>"
            phase0 = obs["phaseID"].get(tls0, -1)
            r0 = rew.get(tls0, 0.0)
            print(f"k={k:03d} t={t:7.1f} tls={tls0} phase={phase0} reward={r0:8.1f} done={done}")

            if done:
                break

        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
