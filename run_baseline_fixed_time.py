"""Run SUMO with fixed-time traffic signal control (baseline) using TraCI."""

import os
import sys
from pathlib import Path
import subprocess
import time


def run_fixed_time_baseline(
    cfg_path: str = "scenarios/grid_3x3_lanes3_dynamic_dense/run.sumo.cfg",
    output_dir: str = "results_baseline",
    gui: bool = False,
) -> bool:
    """
    Run SUMO simulation with default fixed-time control (no AI).
    SUMO automatically uses fixed phase cycling for traffic lights.
    """
    cfg_path = Path(cfg_path)
    
    if not cfg_path.exists():
        print(f"❌ Config not found: {cfg_path}")
        return False
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Prepare SUMO command
    sumo_cmd = "sumo-gui" if gui else "sumo"
    
    # Run from project root, use relative path
    cmd = [
        sumo_cmd,
        "-c", str(cfg_path.relative_to(cfg_path.parent.parent)),  # relative path from project root
        "--no-step-log",
        "--quit-on-end",
    ]
    
    print(f"🚀 Running SUMO fixed-time baseline...")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Expected tripinfo output: {cfg_path.parent}/tripinfos.xml")
    print()
    
    try:
        # Run from project root directory
        result = subprocess.run(cmd, check=True, cwd=str(cfg_path.parent.parent))
        print("✅ SUMO baseline run completed!")
        
        # Check tripinfo output
        tripinfo_path = cfg_path.parent / "tripinfos.xml"
        if tripinfo_path.exists():
            print(f"📊 Tripinfo saved to: {tripinfo_path}")
            return True
        else:
            print(f"⚠️  Expected tripinfo at: {tripinfo_path}")
            return True  # Still success if SUMO ran
            
    except subprocess.CalledProcessError as e:
        print(f"❌ SUMO execution failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ SUMO not found. Install SUMO from: https://sumo.dlr.de/")
        return False


if __name__ == "__main__":
    gui = "--gui" in sys.argv
    run_fixed_time_baseline(gui=gui)
