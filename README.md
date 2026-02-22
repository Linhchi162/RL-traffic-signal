# RL Traffic Signal Control (SUMO) - Kickoff

This is a starter workspace for a graduation project on reinforcement learning for traffic signal control using SUMO.

## Step 1 goal

Set up a reproducible local environment and verify SUMO + TraCI integration works.

## Prerequisites

- Python 3.10+ recommended
- SUMO installed
- `SUMO_HOME` environment variable configured

## Quick start

1) Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

3) Verify SUMO integration:

```powershell
python scripts/check_sumo.py
```

If setup is correct, you should see SUMO binary path and TraCI import success.

## Step 2 goal

Run the bundled RiLSA synthetic intersection scenario (Example 4) and verify you can step the simulation and (optionally) control the traffic light via TraCI.

### Scenario files

- `scenarios/RiLSA_example4/rilsa4.net.xml` (network)
- `scenarios/RiLSA_example4/genroutes.rou.xml` (traffic demand)
- `scenarios/RiLSA_example4/vtypes.add.xml` (vehicle types)
- `scenarios/RiLSA_example4/rilsa4_tls.add.xml` (traffic light program)
- `scenarios/RiLSA_example4/run.sumo.cfg` (SUMO config tying everything together)

### Run directly with SUMO (GUI)

```powershell
sumo-gui -c scenarios\RiLSA_example4\run.sumo.cfg
```

### Run via Python + TraCI (CLI)

```powershell
python scripts/run_rilsa_example4.py --steps 1200
```

### Run via Python + TraCI (GUI)

```powershell
python scripts/run_rilsa_example4.py --gui --steps 1200
```

### Run with a simple controller baseline

This baseline just switches to the next phase periodically (not RL yet).

```powershell
python scripts/run_rilsa_example4.py --control --switch-every 20 --steps 1200
```

## Generate multi-intersection scenarios (grid)

If you need many intersections, generate a grid network (NxM junctions) with SUMO's `netgenerate` and create random traffic demand with `randomTrips.py`.

### Generate a 5x5 grid and run it

```powershell
python scripts/generate_grid_scenario.py --x 5 --y 5 --end 1200 --period 2
sumo-gui -c scenarios\grid_5x5\run.sumo.cfg
```

Notes:
- `--x` and `--y` control how many signalized junctions you get.
- `--period` controls traffic intensity (smaller => more vehicles).
- If you don't have `SUMO_HOME` set, the generator will try to infer it from your PATH, or you can pass `--sumo-home <SUMO_INSTALL_DIR>`.

### Variable (time-varying) demand profile

You can generate a scenario with time-varying demand using `--density-schedule` (vehicles/hour/km):

```powershell
python scripts/generate_grid_scenario.py --x 3 --y 3 --lanes 3 --attach-length 300 --vehicle-class passenger --fringe-factor max --end 3600 \
	--density-schedule "0:900:4,900:1800:12,1800:2700:25,2700:3600:8" \
	--random-depart --random-departpos --random-arrivalpos \
	--out grid_3x3_lanes3_dynamic

sumo-gui -c scenarios\grid_3x3_lanes3_dynamic\run.sumo.cfg
```
