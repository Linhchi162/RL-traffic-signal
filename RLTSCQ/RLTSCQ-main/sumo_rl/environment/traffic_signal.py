"""This module contains the TrafficSignal class, which represents a traffic signal in the simulation."""

import os
import sys
from typing import Callable, List, Union
import traci
import math
import time
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
from gymnasium import spaces


class TrafficSignal:
    """This class represents a Traffic Signal controlling an intersection.

    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    # Observation Space
    The default observation for each traffic signal agent is a vector:

    obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]

    - ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
    - ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
    - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
    - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane

    You can change the observation space by implementing a custom observation class. See :py:class:`sumo_rl.environment.observations.ObservationFunction`.

    # Action Space
    Action space is discrete, corresponding to which green phase is going to be open for the next delta_time seconds.

    # Reward Function
    The default reward function is 'diff-waiting-time'. You can change the reward function by implementing a custom reward function and passing to the constructor of :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    # Default min gap of SUMO (see https://sumo.dlr.de/docs/Simulation/Safety.html). Should this be parameterized?
    MIN_GAP = 2.5

    def __init__(
        self,
        env,
        ts_id: str,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
        enforce_max_green: bool,
        begin_time: int,
        reward_fn: Union[str, Callable, List],
        reward_weights: List[float],
        sumo,
    ):
        """Initializes a TrafficSignal object.

        Args:
            env (SumoEnvironment): The environment this traffic signal belongs to.
            ts_id (str): The id of the traffic signal.
            delta_time (int): The time in seconds between actions.
            yellow_time (int): The time in seconds of the yellow phase.
            min_green (int): The minimum time in seconds of the green phase.
            max_green (int): The maximum time in seconds of the green phase.
            enforce_max_green (bool): If True, the traffic signal will always change phase after max green seconds.
            begin_time (int): The time in seconds when the traffic signal starts operating.
            reward_fn (Union[str, Callable]): The reward function. Can be a string with the name of the reward function or a callable function.
            reward_weights (List[float]): The weights of the reward function.
            sumo (Sumo): The Sumo instance.
        """
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.enforce_max_green = enforce_max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_ts_waiting_time = 0.0
        self.last_reward = None
        self.reward_fn = reward_fn
        self.reward_weights = reward_weights
        self.sumo = sumo
        self.green_extension = 0
        self.enforce_max_green = True
        enforce_max_green = True
        self.phase_durations={}
        self.last_sumo_phase_index = self.sumo.trafficlight.getPhase(self.id)
        self.time_on_current_phase = 0
        self.approach_lanes = self._initialize_approach_lanes()
        # In your agent's __init__ method, add the following attributes:

        # To store max queues from the previous timestep for the R_reduction calculation
        self.last_max_queues = {}

        # --- Reward function parameters ---
        # Normalization factor for reward calculation, representing a 'very large' or 
        # 'maximum possible' queue per approach. This helps keep reward values scaled.
        # Adjust this value based on the maximum expected queue on any single approach.
        self.max_queue_norm_factor =62 

        # Weights for the different reward components (α1 and α2 from the formula)
        self.alpha_reduction = 0.6  # Weight for the queue reduction component (α1) — paper Table II
        self.alpha_absolute = 0.4   # Weight for the absolute queue penalty component (α2) — paper Table II

        if type(self.reward_fn) is list:
            self.reward_dim = len(self.reward_fn)
            self.reward_list = [self._get_reward_fn_from_string(reward_fn) for reward_fn in self.reward_fn]
        else:
            self.reward_dim = 1
            self.reward_list = [self._get_reward_fn_from_string(self.reward_fn)]

        if self.reward_weights is not None:
            self.reward_dim = 1  # Since it will be scalarized

        self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,), dtype=np.float32)

        self.observation_fn = self.env.observation_class(self)

        self._build_phases()

        self.lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id))
        )  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_length = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes + self.out_lanes}

        self.observation_space = self.observation_fn.observation_space()
        #self.action_space = spaces.Discrete(self.num_green_phases)
        ###edit 29 apr
        #######self.action_space = spaces.MultiDiscrete([self.num_green_phases, self.max_green])
        self.action_space = spaces.Discrete(self.max_green)

    def _get_reward_fn_from_string(self, reward_fn):
        if type(reward_fn) is str:
            if reward_fn in TrafficSignal.reward_fns.keys():
  
                #print(TrafficSignal.reward_fns.keys())
                return TrafficSignal.reward_fns[reward_fn]
            else:
                raise NotImplementedError(f"Reward function {reward_fn} not implemented")
        return reward_fn

    def _build_phases(self):
        logic = self.sumo.trafficlight.getAllProgramLogics(self.id)[0]

        # SUMO will follow this exact sequence
        self.sumo.trafficlight.setProgramLogic(self.id, logic)

        # Define green phase indices manually based on your tlLogic:
        self.green_phase_indices = [0, 2, 4, 6]  # E/W straight-left, E/W right, N/S straight-left, N/S right
        self.num_green_phases = len(self.green_phase_indices)

        # Store durations if you want to modify them later
        self.green_durations = [logic.phases[i].duration for i in self.green_phase_indices]

        # Optional: store all phases for inspection
        self.all_phases = logic.phases

    @property
    def time_to_act(self):
        """Returns True if the traffic signal should act in the current step."""
        return self.next_action_time == self.env.sim_step

    def update(self):
        try:
            current_sumo_phase_idx = self.sumo.trafficlight.getPhase(self.id)

            if current_sumo_phase_idx != self.last_sumo_phase_index:
                self.time_on_current_phase = 0
                self.last_sumo_phase_index = current_sumo_phase_idx
            else:

                self.time_on_current_phase += 1

        except traci.TraCIException as e:
             if "connection closed" not in str(e).lower():
                  print(f"Warning: TraCI Error during TS {self.id} update: {e}")
             self.last_sumo_phase_index = -1 # Force re-check next time
             self.time_on_current_phase = 0
    

    def set_next_phase(self, green_extension: int = 0):
        new_phase =self.sumo.trafficlight.getPhase(self.id)
        #print(f"\n--- set_next_phase called at sim_step {self.env.sim_step} ---")
        ##print(f"  Target Env Phase Index: {new_phase}, Raw Green Extension Input: {green_extension}")
        #print(f"  Min/Max Green Constraints: [{self.min_green}, {self.max_green}]")

        new_phase_idx_env =self.sumo.trafficlight.getPhase(self.id)
        #print(new_phase)
        if new_phase % 2!=0:
            self.next_action_time = self.env.sim_step + self.delta_time
            return
        #print("----------")
        # Ensure target phase exists in environment definition
        if new_phase_idx_env >= len(self.all_phases) or new_phase_idx_env < 0:
            #print(f"  ERROR: Invalid new_phase index '{new_phase_idx_env}'. Max index is {len(self.all_phases)-1}.")
            return # Or raise error

        target_phase_state_str = self.all_phases[new_phase_idx_env].state
        #print(f"  Target Phase State String (from env definition): '{target_phase_state_str}'")

        # --- 1. Get Current State from SUMO ---
        try:
            current_logics = self.sumo.trafficlight.getAllProgramLogics(self.id)
            if not current_logics:
                print(f"  ERROR: No program logics found for TL '{self.id}' in SUMO.")
                return
            current_logic = current_logics[0] # Assume modifying the first/default logic
            current_sumo_phase_index = self.sumo.trafficlight.getPhase(self.id)
            current_sumo_state_str = self.sumo.trafficlight.getRedYellowGreenState(self.id)
            current_next_switch_time = self.sumo.trafficlight.getNextSwitch(self.id)

            #print(f"\n  --- State BEFORE Modification ---")
            #print(f"  Current SUMO Phase Index: {current_sumo_phase_index}")
            #print(f"  Current SUMO State String: '{current_sumo_state_str}'")
            #print(f"  Current SUMO Next Switch Time (Absolute): {current_next_switch_time}")
            #print(f"  Current ProgramLogic '{current_logic.programID}':")
            #for i, ph in enumerate(current_logic.phases):
                #print(f"    Phase {i}: duration={ph.duration}, state='{ph.state}'")

        except traci.TraCIException as e:
            print(f"  ERROR: traci exception getting current state: {e}")
            return
        except Exception as e:
            print(f"  ERROR: Unexpected exception getting current state: {e}")
            return

        # --- 2. Calculate New Duration for the Target Phase ---
        # Using green_extension as the *basis* for the new duration, then clamping.
        # RENAME variable for clarity.
        
        new_phase_duration = max(min(float(green_extension), float(self.max_green)), float(self.min_green))
        #print(f"\n  --- Calculation ---")
        #print(f"  Calculated New Duration for target state '{target_phase_state_str}': {new_phase_duration} (clamped from {green_extension})")

        # --- 3. Build the Modified Phase List ---
        modified_phases = []
        found_target_phase_in_logic = False
        for phase in current_logic.phases:
            if phase.state == target_phase_state_str:
                # Found the phase in the current SUMO logic that matches the target state string
                found_target_phase_in_logic = True
                #print(f"    Found matching phase in SUMO logic. Original duration: {phase.duration}. Applying new duration: {new_phase_duration}")
                # Create a NEW Phase object with the modified duration
                modified_phases.append(traci.trafficlight.Phase(new_phase_duration, phase.state, phase.minDur, phase.maxDur))
            else:
                # Keep other phases as they were in the *currently fetched* logic
                modified_phases.append(phase)

        #if not found_target_phase_in_logic:
            #print(f"  WARNING: Target phase state '{target_phase_state_str}' not found in the current SUMO program logic phases. Cannot modify duration.")
            # Decide how to handle this: return, raise error, or proceed without modification?
            # Proceeding might be okay if the intention is just to switch TO this phase,
            # but the duration change won't happen. Let's proceed but the prints show the issue.

        # --- 4. Create and Apply the Modified Logic ---
        try:
            # Use the fetched programID, type, and currentPhaseIndex
            modified_logic = traci.trafficlight.Logic(
                current_logic.programID,
                current_logic.type,
                current_logic.currentPhaseIndex, # Keep the current index unless explicitly changing program
                modified_phases # Use the list with the potentially modified duration
            )

            #print(f"\n  --- Applying Modified Logic ---")
            #print(f"  New Logic to be set (ProgramID: {modified_logic.programID}):")
            #for i, ph in enumerate(modified_logic.phases):
                #print(f"    Phase {i}: duration={ph.duration}, state='{ph.state}' {'<- MODIFIED' if ph.state == target_phase_state_str else ''}")

            # *** THE ACTUAL COMMAND TO CHANGE SUMO'S STATE ***
            self.sumo.trafficlight.setProgramLogic(self.id, modified_logic)
            #print(f"  setProgramLogic called for TL '{self.id}'.")

            # Introduce a very small delay IF needed, sometimes traci needs a moment
            # time.sleep(0.01) # Usually not required, but uncomment for testing if verification fails

        except traci.TraCIException as e:
            #print(f"  ERROR: traci exception during setProgramLogic: {e}")
            return # Stop if setting the logic failed
        except Exception as e:
            #print(f"  ERROR: Unexpected exception during setProgramLogic: {e}")
            return

        # --- 5. Verify the Change in SUMO (Fetch Again) ---
        try:
            # Immediately query SUMO again to see if the change took effect
            time.sleep(0.01) # Brief pause MAY help ensure SUMO processes the change before query
            newly_set_logics = self.sumo.trafficlight.getAllProgramLogics(self.id)
            if not newly_set_logics:
                #print(f"  VERIFICATION ERROR: No logics found after setProgramLogic call.")
                logic_after = None
            else:
                logic_after = newly_set_logics[0] # Assume first logic again

            new_sumo_phase_index = self.sumo.trafficlight.getPhase(self.id)
            new_sumo_state_str = self.sumo.trafficlight.getRedYellowGreenState(self.id)
            new_next_switch_time = self.sumo.trafficlight.getNextSwitch(self.id)

            #print(f"\n  --- State AFTER Modification (Verification) ---")
            #print(f"  Current SUMO Phase Index: {new_sumo_phase_index}")
            #print(f"  Current SUMO State String: '{new_sumo_state_str}'")
            #print(f"  Current SUMO Next Switch Time (Absolute): {new_next_switch_time}")

            if logic_after:
                #print(f"  Current ProgramLogic '{logic_after.programID}' in SUMO:")
                verification_successful = False
                for i, ph in enumerate(logic_after.phases):
                    is_target_phase = (ph.state == target_phase_state_str)
                    modified_marker = ""
                    if is_target_phase:
                        modified_marker = f"<- TARGET PHASE. Expected duration: {new_phase_duration:.2f}"
                        # Compare floating point numbers with tolerance
                        if abs(ph.duration - new_phase_duration) < 0.01:
                            modified_marker += " - VERIFIED!"
                            verification_successful = True
                        else:
                            modified_marker += f" - FAILED VERIFICATION! (Actual: {ph.duration:.2f})"

                    #print(f"    Phase {i}: duration={ph.duration:.2f}, state='{ph.state}' {modified_marker}")
                
                #if not verification_successful and found_target_phase_in_logic:
                    #print(f"  VERIFICATION FAILED: The duration for state '{target_phase_state_str}' in SUMO does not match the intended value {new_phase_duration:.2f}.")
                #elif not found_target_phase_in_logic:
                    #print(f"  VERIFICATION NOTE: Target phase state was not in the original logic, so no duration change was expected or verified.")
                #else:
                    #print(f"  VERIFICATION SUCCESSFUL: Duration change seems reflected in SUMO's program logic.")

            else:
                print(f"  Could not fetch logic after setting for verification.")

        except traci.TraCIException as e:
            print(f"  VERIFICATION ERROR: traci exception checking state after modification: {e}")
        except Exception as e:
            print(f"  VERIFICATION ERROR: Unexpected exception checking state after modification: {e}")


        # --- 6. Update Internal State Variables ---
        # Setting next_action_time based on the fixed delta_time (agent decision interval)
        # This is usually the standard approach.
        self.next_action_time = self.env.sim_step + self.delta_time
        #print(f"\n  --- Updating Internal State ---")
        #print(f"  Setting self.next_action_time = {self.env.sim_step} (current step) + {self.delta_time} (fixed delta) = {self.next_action_time}")

        # Original logic for setting internal state:
        self.green_phase = new_phase_idx_env # Track the *intended* green phase index
        #self.time_since_last_phase_change = 0 # Reset time in phase

        #print(f"  Set self.green_phase = {self.green_phase}")
        #print(f"  Reset self.time_since_last_phase_change = {self.time_since_last_phase_change}")
        #print(f"--- set_next_phase finished ---")
    
    
    def compute_observation(self):
        """Computes the observation of the traffic signal."""
        return self.observation_fn()

    def compute_reward(self) -> Union[float, np.ndarray]:
        """Computes the reward of the traffic signal. If it is a list of rewards, it returns a numpy array."""
        if self.reward_dim == 1:
            self.last_reward = self.reward_list[0](self)
        else:
            self.last_reward = np.array([reward_fn(self) for reward_fn in self.reward_list], dtype=np.float32)
            if self.reward_weights is not None:
                self.last_reward = np.dot(self.last_reward, self.reward_weights)  # Linear combination of rewards

        return self.last_reward

    def _pressure_reward(self):
        return self.get_pressure()

    def _average_speed_reward(self):
        return self.get_average_speed()
    """
    def _queue_reward(self):
        return -self.get_total_queued()
    """
    def _initialize_approach_lanes(self):
        """
        Initializes a mapping from approach direction (N, S, E, W) to a list of
        controlled lane IDs belonging to that approach.
        Uses fixed edge names 'n_t', 's_t', 'e_t', 'w_t' as per problem context.
        """
        approach_lanes_map = {'N': [], 'S': [], 'E': [], 'W': []}
        
        # Define the exact edge IDs for N, S, E, W approaches based on problem's route examples
        target_edge_ids_for_approaches = {
            'N': 'n_t',
            'S': 's_t',
            'E': 'e_t',
            'W': 'w_t'
        }


        controlled_lanes_by_this_tl = self.sumo.trafficlight.getControlledLanes(self.id)
        if not controlled_lanes_by_this_tl:

            return approach_lanes_map 


        edge_to_controlled_lanes = {}
        for lane_id in controlled_lanes_by_this_tl:
            try:

                if lane_id not in self.sumo.lane.getIDList():

                    continue
                edge_id = self.sumo.lane.getEdgeID(lane_id)
                if edge_id not in edge_to_controlled_lanes:
                    edge_to_controlled_lanes[edge_id] = []
                edge_to_controlled_lanes[edge_id].append(lane_id)
            except traci.TraCIException as e:
                print(f"Warning: TraCI error getting edge ID for lane {lane_id} (TS: {self.id}): {e}. Skipping this lane.")
                continue

        for approach_name, target_edge_id in target_edge_ids_for_approaches.items():
            if target_edge_id in edge_to_controlled_lanes:
                # These are the lanes on the target_edge_id that are controlled by *this* TL.
                approach_lanes_map[approach_name].extend(edge_to_controlled_lanes[target_edge_id])
                # Sort lanes for consistent order (e.g., lane_0, lane_1)
                approach_lanes_map[approach_name].sort()
            # else:
                # This is normal if this TS doesn't control lanes on 'n_t', or if 'n_t' isn't an incoming edge here.
                # print(f"Debug: Target edge '{target_edge_id}' for approach '{approach_name}' (TS {self.id}) "
                #       f"not found among its controlled edges or has no controlled lanes on it. "
                #       f"Controlled edges for {self.id}: {list(edge_to_controlled_lanes.keys())}")
        return approach_lanes_map
        
    ### edit - queue reduction + queue absolute 
    def _queue_reward(self):
        """
        Calculates a composite reward based on queue reduction and absolute queue size.

        The reward is defined as:
        rt = α1 * R_reduction + α2 * R_absolute

        Where:
        - R_reduction: Measures the decrease in max queue lengths from the previous step.
                    (Σ (Q_max_j,t-1 - Q_max_j,t)) / (N_approaches * Q_max_norm)
        - R_absolute: Penalizes the current total max queue length.
                    (-Σ Q_max_j,t) / (N_approaches * Q_max_norm)

        Requires state from the previous timestep (self.last_max_queues).
        """
        current_max_queues_per_approach = {}
        num_effective_approaches = 0
        approach_order = ['N', 'S', 'E', 'W']

        # Step 1: Calculate the maximum queue for each approach at the current timestep 
        for approach_key in approach_order:
            lanes_for_this_approach = self.approach_lanes.get(approach_key, [])
            
            max_queue_this_approach = 0
            if not lanes_for_this_approach:
                # This approach does not exist or has no controlled lanes.
                pass
            else:
                num_effective_approaches += 1
                for lane_id in lanes_for_this_approach:
                    try:
                        current_lane_queue = self.sumo.lane.getLastStepHaltingNumber(lane_id)
                        if current_lane_queue > max_queue_this_approach:
                            max_queue_this_approach = current_lane_queue
                    except traci.TraCIException as e:
                        print(f"Warning: Error getting queue for lane {lane_id} (TS: {self.id}, Approach: {approach_key}): {e}. Skipping this lane.")
                        continue
            
            current_max_queues_per_approach[approach_key] = max_queue_this_approach

        # Step 2: Calculate the two reward components.
        # Avoid division by zero if there are no approaches with lanes.
        if num_effective_approaches == 0:
            self.last_max_queues = current_max_queues_per_approach
            return 0.0
        
        sum_of_current_max_queues = 0.0
        sum_of_queue_reduction = 0.0
        
        for approach_key in approach_order:
            current_q = current_max_queues_per_approach.get(approach_key, 0)
            # Retrieve the max queue from the previous step, defaulting to 0.
            last_q = self.last_max_queues.get(approach_key, 0)

            sum_of_current_max_queues += current_q
            sum_of_queue_reduction += (last_q - current_q)

        # Normalization denominator from the formula: N_approaches * Q_max
        denominator = num_effective_approaches * self.max_queue_norm_factor
        if denominator == 0:
            denominator = 1.0 # Failsafe to prevent division by zero

        # R_absolute: Absolute Queue Penalty. This is negative as large queues are bad.
        r_absolute = -sum_of_current_max_queues / denominator

        # R_reduction: Queue Reduction Reward. This is positive if queues have shrunk.
        r_reduction = sum_of_queue_reduction / denominator
        
        # Step 3: Calculate the final weighted reward.
        total_reward = (self.alpha_absolute * r_absolute) + (self.alpha_reduction * r_reduction)

        # Step 4: Update state for the next timestep's calculation.
        # The current queues become the 'last' queues for the next call.
        self.last_max_queues = current_max_queues_per_approach

        # For debugging:
        # print(f"TS {self.id}: R_abs={r_absolute:.3f}, R_red={r_reduction:.3f}, Total Reward={total_reward:.3f}")
        
        return total_reward

    

    def _diff_waiting_time_reward(self):
        ts_wait = sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
        reward = self.last_ts_waiting_time - ts_wait
        self.last_ts_waiting_time = ts_wait
        return reward

    def _observation_fn_default(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        density = self.get_lanes_density()
        queue = self.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def get_accumulated_waiting_time_per_lane(self) -> List[float]:
        """Returns the accumulated waiting time per lane.

        Returns:
            List[float]: List of accumulated waiting time of each intersection lane.
        """
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_average_speed(self) -> float:
        """Returns the average speed normalized by the maximum allowed speed of the vehicles in the intersection.

        Obs: If there are no vehicles in the intersection, it returns 1.0.
        """
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed += self.sumo.vehicle.getSpeed(v) / self.sumo.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def get_pressure(self):
        """Returns the pressure (#veh leaving - #veh approaching) of the intersection."""
        return sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes) - sum(
            self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes
        )

    def get_out_lanes_density(self) -> List[float]:
        """Returns the density of the vehicles in the outgoing lanes of the intersection."""
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.out_lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_density(self) -> List[float]:
        """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        """
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_queue(self) -> List[float]:
        """Returns the queue [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        """
        #lanes_queue = [
        #    self.sumo.lane.getLastStepHaltingNumber(lane)
        #    / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
        #    for lane in self.lanes
        #]
        lanes_queue= [
            self.sumo.lane.getLastStepHaltingNumber(lane)
            for lane in self.lanes
        ]
        #return [min(1, queue) for queue in lanes_queue]
        return [queue for queue in lanes_queue]

    def get_total_queued(self) -> int:
        """Returns the total number of vehicles halting in the intersection."""
        return sum(self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes)

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list

    @classmethod
    def register_reward_fn(cls, fn: Callable):
        """Registers a reward function.

        Args:
            fn (Callable): The reward function to register.
        """
        if fn.__name__ in cls.reward_fns.keys():
            raise KeyError(f"Reward function {fn.__name__} already exists")

        cls.reward_fns[fn.__name__] = fn

    reward_fns = {
        "diff-waiting-time": _diff_waiting_time_reward,
        "average-speed": _average_speed_reward,
        "queue": _queue_reward,
        "pressure": _pressure_reward,
    }


