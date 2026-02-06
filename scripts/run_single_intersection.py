import argparse
import os
import shutil
import sys
from pathlib import Path

import traci


def resolve_sumo_binary(use_gui: bool) -> str:
    binary_name = "sumo-gui" if use_gui else "sumo"
    binary = shutil.which(binary_name)
    if binary:
        return binary

    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise RuntimeError("SUMO_HOME is not set and SUMO binary was not found in PATH.")

    exe = f"{binary_name}.exe" if os.name == "nt" else binary_name
    candidate = os.path.join(sumo_home, "bin", exe)
    if not os.path.exists(candidate):
        raise RuntimeError(f"SUMO binary not found at: {candidate}")
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal single-intersection SUMO episode.")
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui instead of sumo.")
    parser.add_argument("--steps", type=int, default=1800, help="Simulation steps to run.")
    parser.add_argument("--switch-every", type=int, default=20, help="Switch phase period in seconds.")
    parser.add_argument(
        "--control",
        action="store_true",
        help="If set, do simple phase switching (baseline-style controller).",
    )
    args = parser.parse_args()

    scenario_dir = Path(__file__).resolve().parents[1] / "scenarios" / "single_intersection"
    cfg_path = scenario_dir / "single.sumocfg"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    sumo_binary = resolve_sumo_binary(args.gui)
    sumo_cmd = [sumo_binary, "-c", str(cfg_path), "--no-step-log", "true"]

    traci.start(sumo_cmd)
    try:
        tls_ids = traci.trafficlight.getIDList()
        if not tls_ids:
            raise RuntimeError("No traffic lights found in the scenario.")
        tls_id = tls_ids[0]

        controlled_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
        program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        phase_count = len(program.phases)

        print(f"Using TLS id: {tls_id}")
        print(f"Controlled lanes: {controlled_lanes}")
        print(f"Phase count: {phase_count}")

        total_queue = 0.0
        total_wait = 0.0

        for step in range(args.steps):
            if args.control and phase_count >= 2 and step % max(args.switch_every, 1) == 0:
                current_phase = traci.trafficlight.getPhase(tls_id)
                next_phase = (current_phase + 1) % phase_count
                traci.trafficlight.setPhase(tls_id, next_phase)

            traci.simulationStep()

            queue = sum(traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in controlled_lanes)
            wait = sum(traci.lane.getWaitingTime(lane_id) for lane_id in controlled_lanes)
            total_queue += queue
            total_wait += wait

            if step % 300 == 0:
                print(f"step={step:4d} queue={queue:6.2f} waiting={wait:8.2f}")

        avg_queue = total_queue / args.steps
        avg_wait = total_wait / args.steps
        print("\nEpisode done.")
        print(f"Average queue length: {avg_queue:.2f}")
        print(f"Average waiting time: {avg_wait:.2f}")

    finally:
        traci.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
