"""Observation functions for traffic signals."""

from abc import abstractmethod
import logging
import numpy as np
from gymnasium import spaces
import traci 
from .traffic_signal import TrafficSignal
import time
import threading
from numpy import inf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import time

from itertools import combinations
import numpy as np

# Assuming a standard RL environment library like Gymnasium
from gymnasium import spaces
import torch.nn.functional as F

import os
import time
import logging
import threading
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from gymnasium import spaces
import traci

class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)
        self.green_phase_states = {i: phase.state for i, phase in enumerate(self.ts.green_phases)}
        # Store mapping from state string back to green phase index
        self.state_to_green_phase_idx = {state: i for i, state in self.green_phase_states.items()}
   
    def observation_space(self) -> spaces.Box:

        num_phases = self.ts.num_green_phases
        num_lanes = len(self.ts.lanes)

        total_obs_size = num_phases + num_phases + 1 + num_lanes
        low = np.zeros(total_obs_size, dtype=np.float32)
        high_phase_id = np.ones(num_phases, dtype=np.float32)
        high_durations = np.full(num_phases, np.inf, dtype=np.float32) 
        high_remaining = np.full(1, np.inf, dtype=np.float32) 
        high_queue = np.full(num_lanes, np.inf, dtype=np.float32) 
        high = np.concatenate([high_phase_id, high_durations, high_remaining, high_queue])

        return spaces.Box(low=low, high=high, dtype=np.float32)



class MyObservationWithDurationsDirectTraci:
    ## inputs are : current phase id,  all the phase durations, the remaining time in the current phase, queue lenghts at each lane

    def __init__(self, ts):
        self.ts = ts
    def _is_green_phase_state(self, state_str: str) -> bool:

        has_green = 'G' in state_str or 'g' in state_str
        has_yellow = 'y' in state_str or 'Y' in state_str
        return has_green and not has_yellow

    def __call__(self):
        green_phase_states_ordered = []
        all_phase_durations = []
        current_logic = None
        num_green_phases = 0
        try:
            logics = self.ts.sumo.trafficlight.getAllProgramLogics(self.ts.id)
            current_logic = logics[0]
            for phase_in_logic in current_logic.phases:
                if self._is_green_phase_state(phase_in_logic.state):
                    green_phase_states_ordered.append(phase_in_logic.state)
                    all_phase_durations.append(float(phase_in_logic.duration))
            num_green_phases = len(green_phase_states_ordered)
            phase_id = [0.0] * num_green_phases
            current_state_str = self.ts.sumo.trafficlight.getRedYellowGreenState(self.ts.id)
            current_green_index = -1
            if self._is_green_phase_state(current_state_str):
                    current_green_index = green_phase_states_ordered.index(current_state_str)
                    phase_id[current_green_index] = 1.0
            remaining_time = 0.0
            next_switch_time = self.ts.sumo.trafficlight.getNextSwitch(self.ts.id)
            current_sim_time = self.ts.env.sim_step
            remaining_time = max(0.0, next_switch_time - current_sim_time)
            remaining_time_in_phase = [remaining_time]
            queue = self.ts.get_lanes_queue()

            queue = [float(q) for q in queue]
            observation = np.array(
                phase_id + all_phase_durations + remaining_time_in_phase + queue,
                dtype=np.float32
            )
            return observation

        except traci.TraCIException as e:
            print(f"Error during TraCI call in observation for TS {self.ts.id}: {e}. Returning zeros.")
            # Attempt to return a zero array of the correct size if possible
            num_lanes = len(self.ts.lanes) if hasattr(self.ts, 'lanes') else 0
            # Try to get num_green_phases if already calculated, else guess/default
            known_num_green = num_green_phases if num_green_phases > 0 else 2 # Default guess
            obs_len = known_num_green + known_num_green + 1 + num_lanes
            return np.zeros(obs_len, dtype=np.float32)
        except Exception as e:
            print(f"Unexpected error during observation calculation for TS {self.ts.id}: {e}. Returning zeros.")
            # Fallback zero array
            num_lanes = len(self.ts.lanes) if hasattr(self.ts, 'lanes') else 0
            known_num_green = num_green_phases if num_green_phases > 0 else 2 # Default guess
            obs_len = known_num_green + known_num_green + 1 + num_lanes
            return np.zeros(obs_len, dtype=np.float32)


    def observation_space(self):
        """
        Return the Gym observation space definition.
        Determines space size dynamically by querying TraCI for phase info.
        """
        num_green_phases = 0
        try:
            # Query SUMO logic to find the number of green phases dynamically
            logics = self.ts.sumo.trafficlight.getAllProgramLogics(self.ts.id)
            if logics:
                current_logic = logics[0]
                for phase_in_logic in current_logic.phases:
                    if self._is_green_phase_state(phase_in_logic.state):
                        num_green_phases += 1
            else:
                 # Fallback if no logic found - how many phases to assume?
                 # This makes the space definition unstable.
                 print(f"Warning: Cannot determine number of green phases for TS {self.ts.id} space definition. Assuming 2.")
                 num_green_phases = 2 # Or raise an error?

            if num_green_phases == 0:
                 print(f"Warning: No green phases found for TS {self.ts.id} space definition. Assuming 2.")
                 num_green_phases = 2 # Default fallback

        except traci.TraCIException as e:
            print(f"TraCI Error determining observation space size for {self.ts.id}: {e}. Assuming 2 green phases.")
            num_green_phases = 2 # Default fallback
        except Exception as e:
             print(f"Error determining observation space size for {self.ts.id}: {e}. Assuming 2 green phases.")
             num_green_phases = 2


        num_lanes = len(self.ts.lanes) if hasattr(self.ts, 'lanes') else 0
        # phase_id + all_phase_durations + remaining_time + queue
        obs_len = num_green_phases + num_green_phases + 1 + num_lanes

        low = np.zeros(obs_len, dtype=np.float32)
        high = np.ones(obs_len, dtype=np.float32) # Default high to 1

        # Set bounds for durations and remaining time
        # Use max_green or a large fallback value
        max_val = float(self.ts.max_green if hasattr(self.ts, 'max_green') and self.ts.max_green > 0 else 3600.0)
        duration_start_idx = num_green_phases
        duration_end_idx = num_green_phases + num_green_phases
        remaining_time_idx = num_green_phases + num_green_phases

        high[duration_start_idx : duration_end_idx] = max_val # Durations
        high[remaining_time_idx] = max_val # Remaining time

        # Queue bounds are already [0, 1] from normalization in get_lanes_queue

        return spaces.Box(low=low, high=high, dtype=np.float32)
### 19 dimensional state function



class FullStateObservation:
    """
    Generates a 19-dimensional state vector for a traffic signal agent.

    The state vector `st` is composed of:
    st = [Tc, P, tp, cycles, Q_N,S,E,W, ∆Q_N,S,E,W, g_1,2,3,4]

    Where:
    - Tc (1): Normalized total time elapsed in the current cycle.
    - P (4): One-hot encoding of the current active green phase (assumes up to 4 phases).
    - tp (1): Normalized time elapsed in the current phase.
    - cycles (1): Normalized count of completed signal cycles.
    - Qi (4): Normalized max queue length for each approach (N, S, E, W).
    - ∆Qi (4): Normalized change in queue length from the previous step for each approach.
    - gi (4): Normalized programmed green time for each of the first four green phases.
    """

    def __init__(self, ts, max_queue=150.0, max_green_time=120.0):

        self.ts = ts

        self.MAX_QUEUE = float(max_queue)
        self.MAX_GREEN_TIME = float(max_green_time)
        self.MAX_CYCLE_TIME = float(max_green_time * 4) 
        self.MAX_CYCLES_OBS = 10.0  

        self.last_approach_queues = {'N': 0.0, 'S': 0.0, 'E': 0.0, 'W': 0.0}
        self.cycles = 0.0
        self.time_in_current_phase = 0.0
        self.total_cycle_time = 0.0
        self.last_phase_index = -1


        self.green_phases_ordered = []
        self.phase_green_times = []
        self.num_green_phases = 0

        self.approach_lanes = ts.approach_lanes
        self.approach_order = ['N', 'S', 'E', 'W']
        self.num_approaches = len(self.approach_order)
        self.num_programmable_phases = 4 

        self._initialize_phase_info()

    def _initialize_phase_info(self):
        """ Fetches the green phase definitions and their durations from SUMO. """
        try:
            logics = self.ts.sumo.trafficlight.getAllProgramLogics(self.ts.id)
            if not logics:
                raise ValueError("No logic found for traffic light.")
            
            for phase in logics[0].phases:
                if 'g' in phase.state.lower() and 'y' not in phase.state.lower():
                    self.green_phases_ordered.append(phase.state)
                    self.phase_green_times.append(phase.duration)
            
            self.num_green_phases = len(self.green_phases_ordered)
            if self.num_green_phases == 0:
                print(f"Warning: No green phases found for TS {self.ts.id}. Observation may be zero.")

        except (traci.TraCIException, IndexError, ValueError) as e:
            print(f"Error initializing phase info for {self.ts.id}: {e}. Using defaults.")
            # Default to 4 phases if info is not available to maintain vector size
            self.num_green_phases = 0
            self.phase_green_times = []


    def _get_current_phase_index(self):
        """ Returns the index of the current green phase, or -1 if not in a green phase. """
        try:
            state_str = self.ts.sumo.trafficlight.getRedYellowGreenState(self.ts.id)
            if 'y' not in state_str.lower() and 'g' in state_str.lower():
                if state_str in self.green_phases_ordered:
                    return self.green_phases_ordered.index(state_str)
        except traci.TraCIException as e:
            print(f"Error getting phase index for {self.ts.id}: {e}")
        return -1

    def _get_approach_queues(self):
        """ Calculates the maximum queue length for each cardinal approach. """
        queues = {app: 0.0 for app in self.approach_order}
        for approach, lanes in self.approach_lanes.items():
            if not lanes: continue
            
            max_q = max(
                self.ts.sumo.lane.getLastStepHaltingNumber(lane)
                for lane in lanes
            )
            queues[approach] = float(max_q)
        return queues

    def __call__(self):
        """
        Computes and returns the full 19-dimensional state vector.
        """
        #print("hellloooo")
        try:
            current_phase_index = self._get_current_phase_index()
            # Assumes the environment provides the simulation step duration.
            step_duration = self.ts.env.sim_step - getattr(self.ts.env, '_last_step_time', self.ts.env.sim_step)


            if current_phase_index != -1 and current_phase_index != self.last_phase_index:
                # A new green phase has started.

                if self.last_phase_index == self.num_green_phases - 1 and current_phase_index == 0:
                    self.cycles += 1
                    self.total_cycle_time = 0.0  # Reset cycle timer
                
                self.time_in_current_phase = 0.0  # Reset phase timer
                self.last_phase_index = current_phase_index
            
            self.time_in_current_phase += step_duration
            self.total_cycle_time += step_duration
            

            current_queues = self._get_approach_queues()
            q_vec = [current_queues[app] for app in self.approach_order]
            delta_q_vec = [
                current_queues[app] - self.last_approach_queues[app]
                for app in self.approach_order
            ]
            self.last_approach_queues = current_queues


            tc_norm = min(self.total_cycle_time / self.MAX_CYCLE_TIME, 1.0)
            tp_norm = min(self.time_in_current_phase / self.MAX_GREEN_TIME, 1.0)
            cycles_norm = min(self.cycles / self.MAX_CYCLES_OBS, 1.0)

            p_one_hot = [0.0] * self.num_programmable_phases
            if current_phase_index != -1 and current_phase_index < self.num_programmable_phases:
                p_one_hot[current_phase_index] = 1.0

            q_norm = [min(q / self.MAX_QUEUE, 1.0) for q in q_vec]
            delta_q_norm = [np.clip(dq / self.MAX_QUEUE, -1.0, 1.0) for dq in delta_q_vec]
            
            g_norm = [min(g / self.MAX_GREEN_TIME, 1.0) for g in self.phase_green_times]
            g_norm.extend([0.0] * (self.num_programmable_phases - len(g_norm)))
            g_norm = g_norm[:self.num_programmable_phases]

            observation = np.array(
                [tc_norm] + p_one_hot + [tp_norm, cycles_norm] + q_norm + delta_q_norm + g_norm,
                dtype=np.float32
            )
            return observation
       
        except (traci.TraCIException, KeyError, Exception) as e:
            print(f"Error during observation calculation for {self.ts.id}: {e}. Returning zeros.")
            return np.zeros(self.observation_space().shape, dtype=np.float32)

    def observation_space(self):
        """ Returns the Gym observation space for this state vector. """
        # Tc(1) + P(4) + tp(1) + cycles(1) + Q(4) + dQ(4) + g(4) = 19 dimensions
        dims = 1 + self.num_programmable_phases + 1 + 1 + self.num_approaches * 2 + self.num_programmable_phases
        
        low = np.zeros(dims, dtype=np.float32)
        # Set lower bound for ∆Qi to -1
        delta_q_start_index = 1 + self.num_programmable_phases + 2 + self.num_approaches
        low[delta_q_start_index : delta_q_start_index + self.num_approaches] = -1.0

        high = np.ones(dims, dtype=np.float32)
        
        return spaces.Box(low=low, high=high, dtype=np.float32)





class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
    def forward(self, x): return self.model(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x): return self.model(x)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
    def forward(self, x): return self.decoder(self.encoder(x))



class KPlanesStateRepresentation:
    """
    Standalone observation class using a fixed, non-learnable K-Planes transformation.

    This class acts as a sophisticated feature engineering layer that can be used
    directly as an `observation_class` in SumoEnvironment.

    1. It internally uses `FullStateObservation` to get the raw 19D state vector.
    2. It then applies the K-Planes factorization using fixed random planes
       (initialized once) to generate a high-dimensional feature vector.
    3. This feature vector becomes the observation for the RL agent.

    The planes are not trainable, serving as a fixed, powerful non-linear projection.
    """

    def __init__(self, ts, *, feature_dim: int = 16, plane_resolution: int = 32):

        self.ts = ts
        self.feature_dim = feature_dim
        self.plane_resolution = plane_resolution


        self.inner_observation = FullStateObservation(ts)

        # Define the structure of the 19D input vector
        self.groups = {
            'time': {'dims': 3, 'indices': [0, 5, 6]},
            'phase': {'dims': 4, 'indices': [1, 2, 3, 4]},
            'queue': {'dims': 4, 'indices': [7, 8, 9, 10]},
            'delta_q': {'dims': 4, 'indices': [11, 12, 13, 14]},
            'green': {'dims': 4, 'indices': [15, 16, 17, 18]},
        }
        
        # Create the FIXED random planes for each continuous group.
        # These are NOT nn.Parameters. They are just fixed tensors.
        self.planes = {}
        for name, group_info in self.groups.items():
            if name == 'phase':
                continue
            
            num_dims = group_info['dims']
            plane_keys = list(combinations(range(num_dims), 2))
            
            group_planes = {}
            for key in plane_keys:
                # Initialize with random data and keep it fixed
                plane = torch.randn(1, self.feature_dim, self.plane_resolution, self.plane_resolution)
                group_planes[f"{key[0]}_{key[1]}"] = plane
            
            self.planes[name] = group_planes

    def __call__(self) -> np.ndarray:

        # 1. Get the raw 19D state vector from the inner class
        raw_19d_state = self.inner_observation() # This returns a numpy array
        
        # 2. Convert to a torch tensor for processing
        state = torch.from_numpy(raw_19d_state).float().unsqueeze(0) # Add batch dim
        
        # Extract the categorical phase vector
        phase_indices = self.groups['phase']['indices']
        phase_vector = state[:, phase_indices]

        group_features = []
        for name, group_info in self.groups.items():
            if name == 'phase':
                continue

            indices = group_info['indices']
            sub_vector = state[:, indices]
            
            plane_features_for_group = []
            
            for key_str, plane in self.planes[name].items():
                dim1, dim2 = map(int, key_str.split('_'))
                coords = sub_vector[:, [dim1, dim2]]
                
                if name != 'delta_q':
                    grid_coords = coords * 2.0 - 1.0
                else:
                    grid_coords = coords
                    
                grid_coords = grid_coords.unsqueeze(1).unsqueeze(1)
                
                sampled_features = F.grid_sample(
                    plane,  
                    grid_coords,
                    mode='bilinear',
                    align_corners=True,
                    padding_mode='border'
                ).squeeze(-1).squeeze(-1)
                
                plane_features_for_group.append(sampled_features)
            
            combined_group_feature = torch.stack(plane_features_for_group, dim=0).prod(dim=0)
            group_features.append(combined_group_feature)

        final_features_tensor = torch.cat(group_features + [phase_vector], dim=1)

        return final_features_tensor.squeeze(0).detach().numpy()

    @property
    def output_dim(self) -> int:

        num_continuous_groups = len(self.groups) - 1
        return num_continuous_groups * self.feature_dim + self.groups['phase']['dims']

    def observation_space(self) -> spaces.Box:

        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.output_dim,),
            dtype=np.float32
        )





class FullStateObservation:
    """
    Generates a 19-dimensional state vector for a traffic signal agent.
    This serves as the raw input for the autoencoder.
    [Tc, P(4), tp, cycles, Q(4), dQ(4), g(4)]
    """
    def __init__(self, ts, max_queue=150.0, max_green_time=120.0):
        self.ts = ts
        self.MAX_QUEUE = float(max_queue)
        self.MAX_GREEN_TIME = float(max_green_time)
        self.MAX_CYCLE_TIME = float(max_green_time * 4)
        self.MAX_CYCLES_OBS = 10.0
        self.last_approach_queues = {'N': 0.0, 'S': 0.0, 'E': 0.0, 'W': 0.0}
        self.cycles = 0.0
        self.time_in_current_phase = 0.0
        self.total_cycle_time = 0.0
        self.last_phase_index = -1
        self.green_phases_ordered = []
        self.phase_green_times = []
        self.num_green_phases = 0
        self.approach_lanes = getattr(ts, 'approach_lanes', {})
        self.approach_order = ['N', 'S', 'E', 'W']
        self.num_approaches = len(self.approach_order)
        self.num_programmable_phases = 4
        self._initialize_phase_info()

    def _initialize_phase_info(self):
        try:
            logics = self.ts.sumo.trafficlight.getAllProgramLogics(self.ts.id)
            if not logics: return
            for phase in logics[0].phases:
                if 'g' in phase.state.lower() and 'y' not in phase.state.lower():
                    self.green_phases_ordered.append(phase.state)
                    self.phase_green_times.append(phase.duration)
            self.num_green_phases = len(self.green_phases_ordered)
        except (traci.TraCIException, IndexError, ValueError) as e:
            logging.warning(f"Could not initialize phase info for {self.ts.id}: {e}")

    def _get_current_phase_index(self):
        try:
            state_str = self.ts.sumo.trafficlight.getRedYellowGreenState(self.ts.id)
            if 'y' not in state_str.lower() and 'g' in state_str.lower():
                if state_str in self.green_phases_ordered:
                    return self.green_phases_ordered.index(state_str)
        except traci.TraCIException:
            pass
        return -1

    def _get_approach_queues(self):
        queues = {app: 0.0 for app in self.approach_order}
        for approach, lanes in self.approach_lanes.items():
            if not lanes: continue
            try:
                max_q = max(self.ts.sumo.lane.getLastStepHaltingNumber(lane) for lane in lanes)
                queues[approach] = float(max_q)
            except traci.TraCIException:
                pass
        return queues

    def __call__(self):
        try:
            current_phase_index = self._get_current_phase_index()
            step_duration = 1.0

            if current_phase_index != -1 and current_phase_index != self.last_phase_index:
                if self.last_phase_index == self.num_green_phases - 1 and current_phase_index == 0:
                    self.cycles += 1
                    self.total_cycle_time = 0.0
                self.time_in_current_phase = 0.0
                self.last_phase_index = current_phase_index
            
            self.time_in_current_phase += step_duration
            self.total_cycle_time += step_duration
            
            current_queues = self._get_approach_queues()
            q_vec = [current_queues[app] for app in self.approach_order]
            delta_q_vec = [current_queues[app] - self.last_approach_queues[app] for app in self.approach_order]
            self.last_approach_queues = current_queues

            tc_norm = min(self.total_cycle_time / self.MAX_CYCLE_TIME, 1.0)
            tp_norm = min(self.time_in_current_phase / self.MAX_GREEN_TIME, 1.0)
            cycles_norm = min(self.cycles / self.MAX_CYCLES_OBS, 1.0)

            p_one_hot = [0.0] * self.num_programmable_phases
            if current_phase_index != -1 and current_phase_index < self.num_programmable_phases:
                p_one_hot[current_phase_index] = 1.0

            q_norm = [min(q / self.MAX_QUEUE, 1.0) for q in q_vec]
            delta_q_norm = [np.clip(dq / self.MAX_QUEUE, -1.0, 1.0) for dq in delta_q_vec]
            
            g_norm = [min(g / self.MAX_GREEN_TIME, 1.0) for g in self.phase_green_times]
            g_norm.extend([0.0] * (self.num_programmable_phases - len(g_norm)))
            g_norm = g_norm[:self.num_programmable_phases]

            return np.array(
                [tc_norm] + p_one_hot + [tp_norm, cycles_norm] + q_norm + delta_q_norm + g_norm,
                dtype=np.float32
            )
        except (traci.TraCIException, KeyError) as e:
            logging.error(f"Error during observation for {self.ts.id}: {e}")
            return np.zeros(self.observation_space().shape, dtype=np.float32)

    def observation_space(self):
        dims = 1 + self.num_programmable_phases + 1 + 1 + self.num_approaches * 2 + self.num_programmable_phases
        low = np.zeros(dims, dtype=np.float32)
        delta_q_start_index = 1 + self.num_programmable_phases + 2 + self.num_approaches
        low[delta_q_start_index : delta_q_start_index + self.num_approaches] = -1.0
        high = np.ones(dims, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)



class ProjectorEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(ProjectorEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, x): return self.model(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x): return self.model(x)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = ProjectorEncoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
    def forward(self, x): return self.decoder(self.encoder(x))



class _BaseAutoTrainingProjector:

    _training_lock = threading.Lock()
    _is_trained = {}
    _shared_encoder = {}

    def __init__(self, ts, latent_dim: int, buffer_size: int = 10000, epochs: int = 30):
        self.ts = ts
        self.env = ts.env
        self.latent_dim = latent_dim

        self.save_dir = "/seeding_results/autoencoder_dimension_comparison" #### add your save dir for AE training files
        self.model_path = os.path.join(self.save_dir, f"traffic_encoder_{self.latent_dim}d.pth")
        self.log_file = os.path.join(self.save_dir, "autoencoder_performance.csv")
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.input_dim = 19
        self.training_buffer_size = buffer_size
        self.training_epochs = epochs
        self.batch_size = 256
        self.learning_rate = 1e-3
        
        self.base_obs_fn = FullStateObservation(ts)
        self._try_load_model()

    def _try_load_model(self):

        with _BaseAutoTrainingProjector._training_lock:
            if self.latent_dim in _BaseAutoTrainingProjector._is_trained:
                return
            
            if os.path.exists(self.model_path):
                try:
                    print(f"Loading pre-trained model for {self.latent_dim}-D from '{self.model_path}'...")
                    encoder = ProjectorEncoder(self.input_dim, self.latent_dim)
                    encoder.load_state_dict(torch.load(self.model_path))
                    encoder.eval()
                    _BaseAutoTrainingProjector._shared_encoder[self.latent_dim] = encoder
                    _BaseAutoTrainingProjector._is_trained[self.latent_dim] = True
                    print("Model loaded successfully.")
                except Exception as e:
                    logging.warning(f"Could not load model for {self.latent_dim}-D. Error: {e}. Will re-train.")

    def _log_mse_to_file(self, final_mse: float):
        header = "latent_dimension,final_mse\n"
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f: f.write(header)
        with open(self.log_file, 'a') as f: f.write(f"{self.latent_dim},{final_mse:.8f}\n")
        print(f"Logged performance to '{self.log_file}'")

    def _run_training_pipeline(self):

        with _BaseAutoTrainingProjector._training_lock:
            if self.latent_dim in _BaseAutoTrainingProjector._is_trained:
                return

            print("\n" + "="*60)
            print(f"AUTO-TRAINING TRIGGERED FOR {self.latent_dim}-DIMENSIONAL PROJECTOR...")
            
            original_obs_class = self.env.observation_class
            try:
                print("Starting data collection...")
                self.env.observation_class = FullStateObservation
                self.env.reset()
                
                data_buffer = list(self.env._compute_observations().values())
                steps_needed = self.training_buffer_size // len(self.env.ts_ids)
                is_single_agent = not hasattr(self.env, "possible_agents")

                for _ in range(1, steps_needed):
                    action = self.env.action_space.sample() if is_single_agent else {ts_id: self.env.action_space(ts_id).sample() for ts_id in self.env.ts_ids}
                    obs, _, _, _, _ = self.env.step(action)
                    if is_single_agent: data_buffer.append(obs)
                    else: data_buffer.extend(list(obs.values()))

                print(f"Data collection complete. Training on {len(data_buffer)} samples.")

                start_time = time.time()
                dataset = TensorDataset(torch.from_numpy(np.array(data_buffer)).float())
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                autoencoder = Autoencoder(self.input_dim, self.latent_dim)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(autoencoder.parameters(), lr=self.learning_rate)
                autoencoder.train()
                for epoch in range(self.training_epochs):
                    for data in dataloader:
                        inputs = data[0]; optimizer.zero_grad(); outputs = autoencoder(inputs); loss = criterion(outputs, inputs); loss.backward(); optimizer.step()
                autoencoder.eval()
                final_mse_loss = 0.0
                with torch.no_grad():
                    for data in dataloader:
                        inputs = data[0]; reconstructed = autoencoder(inputs); loss = criterion(reconstructed, inputs); final_mse_loss += loss.item()
                average_final_mse = final_mse_loss / len(dataloader)
                print(f"  >> Final Average Reconstruction MSE: {average_final_mse:.8f} <<")
                self._log_mse_to_file(average_final_mse)
                duration = time.time() - start_time
                print(f"Training complete in {duration:.2f} seconds.")

                # Finalize and save
                autoencoder.encoder.eval()
                _BaseAutoTrainingProjector._shared_encoder[self.latent_dim] = autoencoder.encoder
                _BaseAutoTrainingProjector._is_trained[self.latent_dim] = True
                torch.save(autoencoder.encoder.state_dict(), self.model_path)
                print(f"Encoder model saved to '{self.model_path}'")

            finally:
                print("Restoring original observation class attribute...")
                self.env.observation_class = original_obs_class

            print("AUTO-TRAINING LOGIC COMPLETE.")
            print("="*60 + "\n")

    def __call__(self) -> np.ndarray:

        if self.latent_dim not in _BaseAutoTrainingProjector._is_trained:

            self._run_training_pipeline()
            for ts_id in self.env.ts_ids:

                self.env.traffic_signals[ts_id].observation_fn = self.env.observation_class(
                    self.env.traffic_signals[ts_id]
                )

            return np.zeros(self.observation_space().shape, dtype=np.float32)

        return self.get_features()

    def get_features(self) -> np.ndarray:
        raw_obs = self.base_obs_fn()
        obs_tensor = torch.from_numpy(raw_obs).float().unsqueeze(0)
        with torch.no_grad():
            encoder = _BaseAutoTrainingProjector._shared_encoder[self.latent_dim]
            features = encoder(obs_tensor)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.latent_dim,), dtype=np.float32)

# FINAL OBSERVATION CLASSES FOR EXPERIMENTS 
class AutoTrainingProjector_Latent_4(_BaseAutoTrainingProjector):
    def __init__(self, ts):
        super().__init__(ts, latent_dim=4)
class AutoTrainingProjector_Latent_8(_BaseAutoTrainingProjector):
    def __init__(self, ts):
        super().__init__(ts, latent_dim=8)
class AutoTrainingProjector_Latent_16(_BaseAutoTrainingProjector):
    def __init__(self, ts):
        super().__init__(ts, latent_dim=16)
class AutoTrainingProjector_Latent_19(_BaseAutoTrainingProjector):
    def __init__(self, ts):
        super().__init__(ts, latent_dim=19)
class AutoTrainingProjector_Latent_32(_BaseAutoTrainingProjector):
    def __init__(self, ts):
        super().__init__(ts, latent_dim=32)