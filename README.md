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

Run a minimal single-intersection simulation and collect basic RL-friendly metrics (queue and waiting time).

### Scenario files

- `scenarios/single_intersection/single.net.xml`
- `scenarios/single_intersection/single.rou.xml`
- `scenarios/single_intersection/single.sumocfg`

### Run one episode (CLI)

```powershell
python scripts/run_single_intersection.py --steps 1200
```

### Run with GUI

```powershell
python scripts/run_single_intersection.py --gui --steps 1200
```

### Run with simple controller baseline

This baseline just switches phase periodically (not RL yet).

```powershell
python scripts/run_single_intersection.py --control --switch-every 20 --steps 1200
```
