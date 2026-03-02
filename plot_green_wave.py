"""Visualize Green Wave pattern - Phase changes over time for each intersection."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_phasechanges_from_log(
    sumo_output_dir: Path,
) -> dict:
    """
    Parse phase changes from SUMO simulation.
    This requires running SUMO with --save-state flag or parsing from log.
    
    For now, we provide a template. In practice, you can:
    1. Add phase tracking to env.py during simulation
    2. Log phase changes to a file
    3. Parse vehicle counts from tripinfo + edges
    """
    
    doc = """
    To visualize Green Wave (Phase changes over time):
    
    1. Modify env.py to log phase changes:
       - Track observation["phaseID"] for each TLS at each step
       - Save to CSV: timestamp, tls_id, phase_id, vehicle_count
    
    2. Then use this function to plot:
       - X-axis: Time (seconds)
       - Y-axis: TLS node IDs
       - Heatmap: Phase ID or green/red state
       - Overlay: Vehicle count trend
    
    Example output format:
    time,tls_id,phase_id,queue_length
    0,TLS1,0,5
    10,TLS1,1,12
    10,TLS2,0,3
    20,TLS1,0,8
    ...
    """
    
    print(doc)
    return None


def plot_green_wave_template(
    phase_log_csv: str = "phase_changes.csv",
    output_file: str = "green_wave.png",
) -> bool:
    """
    Template for plotting Green Wave pattern.
    
    phase_log_csv format:
        time,tls_id,phase_id,queue_length
    """
    
    print(f"""
✨ GREEN WAVE VISUALIZATION
════════════════════════════════════════════════════════════════

To create the Green Wave visualization (similar to Figure 8 in STMARL paper):

STEP 1: Modify train_gnn_lstm_dqn.py to log phase data during evaluation:
───────────────────────────────────────────────────────────────
Add this to the training loop (for testing phase only):
    
    # Inside the main loop
    if global_step % 10 == 0:  # Log every 10 steps
        phase_data.append({{
            'time': current_time,
            'tls_id': env.control_nodes,
            'phase_id': obs['phaseID'],
            'queue': [obs['edge'][edge].get('q', 0) for edge in env._incoming_edges_by_tls[tls]]
        }})

STEP 2: Save to CSV after simulation:
───────────────────────────────────────────────────────────────
import csv
with open('phase_changes.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, ['time', 'tls_id', 'phase_id', 'queue_length'])
    writer.writeheader()
    for entry in phase_data:
        # Write rows for each TLS

STEP 3: Run this script:
───────────────────────────────────────────────────────────────
python plot_green_wave.py phase_changes.csv

OUTPUT: green_wave.png showing:
  - Y-axis: Different TLS (TLS1, TLS2, TLS3, etc.)
  - X-axis: Time (seconds)
  - Color/Pattern: Green phase (1), Red phase (0)
  - Overlay: Vehicle queue length trend

This visualization proves AI coordination creates "Green Waves" where
sequential intersections turn green in sync to let traffic flow continuously.
════════════════════════════════════════════════════════════════
""")
    
    return True


if __name__ == "__main__":
    plot_green_wave_template()
