import os
import sys
import argparse
import gymnasium as gym
from stable_baselines3 import PPO 
import csv
import numpy as np
import time
import collections

LOW_SPEED_THRESHOLD = 0.1

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

try:
    import traci
    from sumo_rl import SumoEnvironment
except ImportError as e:
    sys.exit(f"Error importing SUMO-related libraries: {e}. Check SUMO_HOME and sumo_rl installation.")
except FileNotFoundError as e:
    sys.exit(f"Error: sumo_rl environment file not found. Ensure it's installed correctly: {e}")




def get_low_speed_vehicles_per_lane(lane_ids):
    low_speed_counts = {}
    if not lane_ids:
        return low_speed_counts
    threshold = LOW_SPEED_THRESHOLD
    try:
        existing_lanes = set(traci.lane.getIDList())
        for lane_id in lane_ids:
            if lane_id in existing_lanes:
                current_lane_low_speed_count = 0
                try:
                    vehicle_ids_on_lane = traci.lane.getLastStepVehicleIDs(lane_id)
                    for vehID in vehicle_ids_on_lane:
                        try:
                            speed = traci.vehicle.getSpeed(vehID)
                            if speed < threshold:
                                current_lane_low_speed_count += 1
                        except traci.TraCIException: # Vehicle might have left
                            pass
                        except Exception as e_inner:
                            print(f"Warning: Unexpected error getting speed for {vehID} on lane {lane_id}: {e_inner}")
                            pass
                    low_speed_counts[lane_id] = current_lane_low_speed_count
                except traci.TraCIException as e_lane:
                    if "connection closed" not in str(e_lane).lower():
                        print(f"Warning: traci call failed getting vehicle IDs for lane {lane_id}: {e_lane}")
                    low_speed_counts[lane_id] = 0
                except Exception as e_outer_lane:
                    print(f"Warning: Unexpected error processing lane {lane_id}: {e_outer_lane}")
                    low_speed_counts[lane_id] = 0
            else:
                low_speed_counts[lane_id] = 0
    except traci.TraCIException as e_outer:
        if "connection closed" not in str(e_outer).lower():
            print(f"Warning: Outer traci call failed during low speed vehicle check: {e_outer}")
    except Exception as e:
        print(f"Warning: Unexpected error getting low speed vehicles: {e}")
    return low_speed_counts



if __name__ == "__main__":

    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="PPO Model Testing Script - Logging Max Queues per Cycle", # Updated description
    )
    prs.add_argument(
        "-model",
        dest="model_path",
        type=str,
        default = "/q_ae_Kplanes/ppo_sumo_final_model.zip",
        help="Path to the trained PPO model (.zip file)",
    )
    prs.add_argument(
        "-outputdir",
        dest="output_dir",
        type=str,
        default="/q_ae_Kplanes/test_dir", # Default to a 'test_output' subdirectory
        help="Directory to save test logs",
    )
    
    prs.add_argument("-net", dest="net_file", type=str, default = "nets/RLQ/caliberated_net.xml", help="Path to the SUMO network file (.net.xml)")
    prs.add_argument("-route", dest="route_file", type=str, default =  "nets/RLQ/test_flows.xml", help="Path to the SUMO route file (.rou.xml)")
    prs.add_argument("-testseconds", dest="test_seconds", type=int, default=5000, help="Duration (simulation seconds) for the testing phase")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with SUMO GUI")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, help="Minimum green time used by env (important for context if actions modify time)")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=40, help="Maximum green time used by env (needs to match training for action space)") # Added max_green arg
    prs.add_argument("-yellowtime", dest="yellow_time", type=int, default=5, help="Yellow time duration") # Added yellow_time arg
    prs.add_argument("-deltatime", dest="delta_time", type=int, default=5, help="Simulation seconds between checking if action is needed") # Adjusted help text

    args = prs.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    timestep_log_path = os.path.join(args.output_dir, "test_timestep_log.csv")
    cycle_log_path = os.path.join(args.output_dir, "test_cycle_log.csv")
    test_env_metrics_path = os.path.join(args.output_dir, "test_environment_metrics.csv")

    print("=" * 10 + " Starting Testing Phase (Max Queue Logging) " + "=" * 10)
    print(f"Using manual speed check with threshold: {LOW_SPEED_THRESHOLD} m/s")
    print(f"Loading model from: {args.model_path}")
    print(f"Network: {args.net_file}")
    print(f"Route: {args.route_file}")
    print(f"Test Duration: {args.test_seconds} seconds")
    print(f"Output Logs: {args.output_dir}")
    print(f"Min Green: {args.min_green}, Max Green: {args.max_green}, Delta Time: {args.delta_time}, Yellow Time: {args.yellow_time}")

    try:
        model = PPO.load(args.model_path, env=None)
        print("Model loaded successfully.")
    except Exception as e:
        sys.exit(f"Error loading model from {args.model_path}: {e}")

    sumo_running = False
    try:
        test_env = SumoEnvironment(
            net_file=args.net_file,
            route_file=args.route_file,
            out_csv_name=test_env_metrics_path,
            use_gui=args.gui,
            num_seconds=args.test_seconds + 200,
            min_green=args.min_green,
            max_green=args.max_green,
            yellow_time=args.yellow_time,
            delta_time=args.delta_time,
            single_agent=True,
            #reward_fn="average-speed"
            reward_fn="queue"
        )

        print("Resetting environment and starting SUMO...")
        obs, info = test_env.reset()
        sumo_running = True
        print("SUMO connection established.")

        if not test_env.single_agent or len(test_env.ts_ids) > 1:
             print("Warning: This script assumes a single traffic signal...")

        ts_id = test_env.ts_ids[0]

        all_program_logics = traci.trafficlight.getAllProgramLogics(ts_id)
        if not all_program_logics: sys.exit(f"Error: No program logic found for TS {ts_id}")
        active_program_logic = all_program_logics[0]
        sumo_phases = active_program_logic.phases
        green_phase_indices_in_sumo = []
        for i, phase in enumerate(sumo_phases):
            state = phase.state.lower()
            is_green_char = 'g' in state
            is_yellow_char = 'y' in state
            is_all_red = all(c == 'r' or c == 's' for c in state)
            if is_green_char and not is_yellow_char and not is_all_red:
                 green_phase_indices_in_sumo.append(i)

        if not green_phase_indices_in_sumo: sys.exit(f"Error: No green phases identified for {ts_id}")
        num_green_phases = len(green_phase_indices_in_sumo)
        first_green_phase_index_in_sumo = green_phase_indices_in_sumo[0]
        incoming_lanes = sorted(list(set(traci.trafficlight.getControlledLanes(ts_id)))) # Sort here
        controlled_lanes_by_green_phase = {}
        try: links = traci.trafficlight.getControlledLinks(ts_id)
        except traci.TraCIException as e: sys.exit(f"Error getting controlled links for {ts_id}: {e}")
        for idx in green_phase_indices_in_sumo:
            controlled_lanes_by_green_phase[idx] = []
            state_str = sumo_phases[idx].state
            for link_tuple in links:
                 if isinstance(link_tuple, tuple) and len(link_tuple) >= 2 and \
                    isinstance(link_tuple[0], tuple) and len(link_tuple[0]) >= 1:
                     in_lane = link_tuple[0][0]
                     link_index_in_tl = link_tuple[1]
                     if link_index_in_tl < len(state_str):
                         if state_str[link_index_in_tl].lower() == 'g':
                             controlled_lanes_by_green_phase[idx].append(in_lane)
                     else: print(f"Warning: Link index {link_index_in_tl} out of bounds...")
                 else: print(f"Warning: Unexpected link tuple format...")
            controlled_lanes_by_green_phase[idx] = sorted(list(set(controlled_lanes_by_green_phase[idx]))) # Sort lanes within phase

        print(f"Identified Traffic Signal: {ts_id}")
        print(f"Total SUMO Phases: {len(sumo_phases)}")
        print(f"Green Phase Indices in SUMO logic: {green_phase_indices_in_sumo}")
        print(f"Associated Incoming Lanes: {incoming_lanes}")


    except traci.TraCIException as e: sys.exit(f"Error connecting to SUMO/getting info: {e}")
    except Exception as e: import traceback; traceback.print_exc(); sys.exit(f"Error initializing: {e}")

    ts_fieldnames = [
        'sim_time', 'sumo_phase_index', 'phase_state_str', 'time_in_phase', 'time_until_next_switch',
        'action_phase_choice', 'action_green_extension',
        'current_low_speed_vehicles_total' # Renamed for clarity vs max
    ]
    for lane in incoming_lanes: # Use already sorted list
        ts_fieldnames.append(f'current_low_speed_{lane}') # Renamed

    try:
        ts_csv_file = open(timestep_log_path, 'w', newline='')
        ts_writer = csv.DictWriter(ts_csv_file, fieldnames=ts_fieldnames)
        ts_writer.writeheader()
    except IOError as e: sys.exit(f"Error opening timestep log file {timestep_log_path}: {e}")

    cy_fieldnames = [
        'cycle_number', 'cycle_start_time', 'cycle_end_time', 'total_cycle_time'
    ]
    for phase_idx in sorted(green_phase_indices_in_sumo):
        cy_fieldnames.append(f'green_time_phase_{phase_idx}')
    for phase_idx in sorted(green_phase_indices_in_sumo):
        cy_fieldnames.append(f'max_queue_during_red_phase_{phase_idx}')

    for lane_id in incoming_lanes:
        cy_fieldnames.append(f'max_queue_lane_{lane_id}')

    try:
        cy_csv_file = open(cycle_log_path, 'w', newline='')
        cy_writer = csv.DictWriter(cy_csv_file, fieldnames=cy_fieldnames)
        cy_writer.writeheader()
    except IOError as e: sys.exit(f"Error opening cycle log file {cycle_log_path}: {e}")

    done = False
    sim_step_count = 0
    loop_iterations = 0
    start_test_time = time.time()
    last_action = (-1, -1)

    current_cycle_number = 0
    cycle_start_sim_time = 0.0
    time_current_phase_started = 0.0
    phase_green_times_in_cycle = collections.defaultdict(float)
    green_phases_seen_in_cycle = set()
    previous_sumo_phase_index = -1
    current_sim_time = 0.0

    max_queue_per_lane_in_cycle = collections.defaultdict(int)
    max_phase_queue_during_red_in_cycle = collections.defaultdict(int)

    try:

        if sumo_running:
            current_sim_time = traci.simulation.getTime()
            previous_sumo_phase_index = traci.trafficlight.getPhase(ts_id)
            time_current_phase_started = current_sim_time

            initial_queues = get_low_speed_vehicles_per_lane(incoming_lanes)
            for lane, count in initial_queues.items():
                max_queue_per_lane_in_cycle[lane] = count

            initial_phase_index = previous_sumo_phase_index
            for p_idx in green_phase_indices_in_sumo:
                 if initial_phase_index != p_idx: # If phase starts as red
                      lanes_for_phase = controlled_lanes_by_green_phase.get(p_idx, [])
                      current_phase_q = sum(initial_queues.get(l, 0) for l in lanes_for_phase)
                      max_phase_queue_during_red_in_cycle[p_idx] = current_phase_q
        else:
             raise RuntimeError("SUMO failed to start...")

        while current_sim_time < args.test_seconds:
            loop_iterations += 1


            current_sim_time = traci.simulation.getTime()
            current_sumo_phase_index = traci.trafficlight.getPhase(ts_id)

            current_phase_state_str = traci.trafficlight.getRedYellowGreenState(ts_id)
            time_until_next_switch_abs = traci.trafficlight.getNextSwitch(ts_id)
            if time_until_next_switch_abs < 0: time_until_next_switch_rel = float('inf')
            else: time_until_next_switch_rel = max(0.0, time_until_next_switch_abs - current_sim_time)
            if current_sumo_phase_index != previous_sumo_phase_index: time_in_phase = 0.0
            else: time_in_phase = current_sim_time - time_current_phase_started



            action_phase_choice, action_green_extension = (-1, -1)
            agent_requested_action = False
            action = None
            try:
                 if test_env.traffic_signals[ts_id].time_to_act:
                     agent_requested_action = True
                     action, _states = model.predict(obs, deterministic=True)
                     action_green_extension = int(action)
                     last_action = (action_green_extension)
            except (KeyError, AttributeError) as e:
                 print(f"Warning: Error checking time_to_act: {e}")



            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            sim_step_count += 1


            current_sim_time_after_step = traci.simulation.getTime()
            current_sumo_phase_index_after_step = traci.trafficlight.getPhase(ts_id)
            current_phase_state_str_after_step = traci.trafficlight.getRedYellowGreenState(ts_id)

            time_until_next_switch_abs_after_step = traci.trafficlight.getNextSwitch(ts_id)
            if time_until_next_switch_abs_after_step < 0: time_until_next_switch_rel_after_step = float('inf')
            else: time_until_next_switch_rel_after_step = max(0.0, time_until_next_switch_abs_after_step - current_sim_time_after_step)

            phase_changed_during_step = (current_sumo_phase_index_after_step != current_sumo_phase_index)
            final_sumo_phase_index_this_log = current_sumo_phase_index_after_step

            if final_sumo_phase_index_this_log != previous_sumo_phase_index: time_in_phase_this_log = 0.0
            else: time_in_phase_this_log = current_sim_time_after_step - time_current_phase_started


            current_low_speed_data = get_low_speed_vehicles_per_lane(incoming_lanes)
            current_total_low_speed = sum(current_low_speed_data.values())

            for lane, count in current_low_speed_data.items():
                max_queue_per_lane_in_cycle[lane] = max(max_queue_per_lane_in_cycle[lane], count)

            for p_idx in green_phase_indices_in_sumo:

                if final_sumo_phase_index_this_log != p_idx:
                    lanes_for_phase = controlled_lanes_by_green_phase.get(p_idx, [])
                    if lanes_for_phase: # Ensure the phase has lanes mapped
                        current_phase_queue_sum = sum(current_low_speed_data.get(l, 0) for l in lanes_for_phase)
                        max_phase_queue_during_red_in_cycle[p_idx] = max(
                            max_phase_queue_during_red_in_cycle[p_idx],
                            current_phase_queue_sum
                        )



            ts_row = {
                'sim_time': round(current_sim_time_after_step, 2),
                'sumo_phase_index': final_sumo_phase_index_this_log,
                'phase_state_str': current_phase_state_str_after_step,
                'time_in_phase': round(time_in_phase_this_log, 2),
                'time_until_next_switch': round(time_until_next_switch_rel_after_step, 2) if time_until_next_switch_rel_after_step != float('inf') else -1,
                'action_green_extension': last_action if agent_requested_action else -1,
                'current_low_speed_vehicles_total': current_total_low_speed # Use current total
            }
            for lane in incoming_lanes: # Use sorted list
                # Use current data for timestep log
                ts_row[f'current_low_speed_{lane}'] = current_low_speed_data.get(lane, 0)

            ts_writer.writerow(ts_row)
            if agent_requested_action:
                last_action = (-1, -1) # Reset after logging


            # Check for phase change to potentially log cycle end / update cycle state
            if final_sumo_phase_index_this_log != previous_sumo_phase_index:
                # Calculate duration of the *previous* phase
                previous_phase_duration = current_sim_time_after_step - time_current_phase_started

                # If previous phase was green, add its duration to cycle total & mark seen
                if previous_sumo_phase_index in green_phase_indices_in_sumo:
                    phase_green_times_in_cycle[previous_sumo_phase_index] += previous_phase_duration
                    green_phases_seen_in_cycle.add(previous_sumo_phase_index)

                is_first_green_phase = (final_sumo_phase_index_this_log == first_green_phase_index_in_sumo)
                cycle_started = (cycle_start_sim_time >= 0 and current_cycle_number >= 0 and current_sim_time_after_step > 0)
                # Ensure all unique green phases expected have been seen
                all_greens_seen = (len(green_phases_seen_in_cycle) >= num_green_phases)

                if is_first_green_phase and cycle_started and all_greens_seen:
                    current_cycle_number += 1
                    cycle_end_sim_time = current_sim_time_after_step # Cycle ends as the new one begins
                    total_cycle_time = cycle_end_sim_time - cycle_start_sim_time

                    if total_cycle_time < args.yellow_time: # Basic check
                         print(f"Warning: Short cycle time ({total_cycle_time:.2f}s) at cycle {current_cycle_number}.")

                    cy_row = {
                        'cycle_number': current_cycle_number,
                        'cycle_start_time': round(cycle_start_sim_time, 2),
                        'cycle_end_time': round(cycle_end_sim_time, 2),
                        'total_cycle_time': round(total_cycle_time, 2),
                    }
                    # Add green times
                    for phase_idx in sorted(green_phase_indices_in_sumo):
                        cy_row[f'green_time_phase_{phase_idx}'] = round(phase_green_times_in_cycle.get(phase_idx, 0.0), 2)
                    # Add MAX phase queues (during red)
                    for phase_idx in sorted(green_phase_indices_in_sumo):
                        cy_row[f'max_queue_during_red_phase_{phase_idx}'] = max_phase_queue_during_red_in_cycle.get(phase_idx, 0) # Get tracked max
                    # Add MAX lane queues (overall)
                    for lane_id in incoming_lanes: # Use sorted list
                         cy_row[f'max_queue_lane_{lane_id}'] = max_queue_per_lane_in_cycle.get(lane_id, 0) # Get tracked max

                    cy_writer.writerow(cy_row)


                    cycle_start_sim_time = current_sim_time_after_step # Start time of the new cycle is now
                    phase_green_times_in_cycle.clear()
                    green_phases_seen_in_cycle.clear()
                    # *** RESET MAX QUEUE TRACKERS ***
                    max_queue_per_lane_in_cycle.clear()
                    max_phase_queue_during_red_in_cycle.clear()
                    # Re-initialize max queues based on current state for the start of the new cycle
                    current_queues_new_cycle = get_low_speed_vehicles_per_lane(incoming_lanes)
                    for lane, count in current_queues_new_cycle.items():
                         max_queue_per_lane_in_cycle[lane] = count # Start tracking from current count
                    current_phase_new_cycle = final_sumo_phase_index_this_log
                    for p_idx in green_phase_indices_in_sumo:
                        if current_phase_new_cycle != p_idx:
                            lanes_for_phase = controlled_lanes_by_green_phase.get(p_idx, [])
                            current_phase_q = sum(current_queues_new_cycle.get(l, 0) for l in lanes_for_phase)
                            max_phase_queue_during_red_in_cycle[p_idx] = current_phase_q


                time_current_phase_started = current_sim_time_after_step

            previous_sumo_phase_index = final_sumo_phase_index_this_log



            if done:
                print(f"Environment terminated or truncated at sim time {current_sim_time_after_step:.2f}. Stopping.")
                break


    except KeyboardInterrupt: print("Testing interrupted by user.")
    except traci.TraCIException as e: print(f"TraCI error during loop: {e}")
    except Exception as e: import traceback; traceback.print_exc(); print(f"Unexpected error during testing: {e}")
    finally:

        end_test_time = time.time()
        print(f"\nTesting finished in {end_test_time - start_test_time:.2f} seconds (real time).")
        final_sim_time = current_sim_time_after_step if 'current_sim_time_after_step' in locals() else 0.0
        print(f"Simulated {sim_step_count} environment steps ({loop_iterations} loop iterations) up to {final_sim_time:.2f} simulation seconds.")
        try:
            if 'ts_writer' in locals() and ts_csv_file and not ts_csv_file.closed:
                ts_csv_file.close(); print(f"Time-step log saved to: {timestep_log_path}")
        except Exception as e: print(f"Error closing timestep log: {e}")
        try:
            if 'cy_writer' in locals() and cy_csv_file and not cy_csv_file.closed:
                cy_csv_file.close(); print(f"Cycle log saved to: {cycle_log_path}")
        except Exception as e: print(f"Error closing cycle log: {e}")
        try:
            if 'test_env' in locals() and hasattr(test_env, 'sumo') and test_env.sumo is not None:
                 print("Attempting to close SumoEnvironment..."); test_env.close(); print("SUMO Env closed.")
                 sumo_running = False
            elif sumo_running: print("Attempting emergency traci close..."); traci.close(); sumo_running = False; print("Direct traci closed.")
        except traci.TraCIException as e: print(f"TraCI error during close: {e}.")
        except Exception as e: print(f"Error closing SUMO: {e}")

    print("Testing script finished.")