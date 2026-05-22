# Reinforcement Learning-based Traffic Signal Control Optimization in SUMO

Bachelor's thesis — University of Engineering and Technology, Vietnam National University, Hanoi.

This repository implements an **Independent Multi-Agent Reinforcement Learning** system for traffic signal control, where each signalised intersection is managed by an independent agent using a shared policy. Three algorithms (PPO, DQN, DDQN) are combined with three reward functions (queue, pressure, wait-clip) and evaluated on two sub-scenarios extracted from the real-world **TAPAS Cologne** dataset.

---

## Results summary

| Scenario | Best config | Mean queue vs Fixed-time | Throughput vs Fixed-time |
|---|---|---|---|
| Cologne-8 (8 intersections) | PPO-pressure | −88 % | +0.8 % |
| Cologne-3 (3 intersections) | DQN-pressure | −80 % | +0.8 % |

All RL configurations outperform both Fixed-time and Webster baselines on Cologne-8. Results on Cologne-3 are less consistent due to sparser learning signal.

---

## Repository structure

```
.
├── rl_controller/
│   ├── traffic_env.py      # Single-intersection Gymnasium environment (TraCI/libsumo)
│   ├── grid_env.py         # MultiAgentVecEnv wrapper (parameter sharing)
│   ├── state_builder.py    # 7-dimensional observation builder
│   ├── intersection.py     # Intersection abstraction (phases, queues, rewards)
│   └── webster.py          # Webster adaptive baseline controller
├── nets/
│   ├── cologne3/           # Cologne-3 network + route files (real + synthetic)
│   └── cologne8/           # Cologne-8 network + route files (real + synthetic)
├── train_cologne_ppo.py    # Train PPO / IPPO
├── train_cologne_dqn.py    # Train DQN / DDQN
├── evaluate_all.py         # Batch evaluation across all configs and seeds
├── evaluate_cologne3_real.py
├── evaluate_cologne8.py
├── plot_learning_curves.py # Plot training reward curves
├── plot_results.py         # Plot evaluation bar charts
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.10+, SUMO installed, `SUMO_HOME` environment variable set.

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Training

### PPO

```bash
python train_cologne_ppo.py \
    --net_file   nets/cologne8/cologne8.net.xml \
    --route_file nets/cologne8/cologne8_rand_100pct.rou.xml \
    --reward_type pressure \
    --seed 42 \
    --total_steps 500000 \
    --save_dir ./exp_cologne8/ppo_pressure_s42
```

### DQN / DDQN

```bash
python train_cologne_dqn.py \
    --net_file   nets/cologne8/cologne8.net.xml \
    --route_file nets/cologne8/cologne8_rand_100pct.rou.xml \
    --algo ddqn \
    --reward_type queue \
    --seed 42 \
    --total_steps 500000 \
    --save_dir ./exp_cologne8/ddqn_queue_s42
```

**Key arguments**

| Argument | Default | Description |
|---|---|---|
| `--net_file` | required | Path to `.net.xml` |
| `--route_file` | required | Path to `.rou.xml` (use `*_rand_*` files for training) |
| `--reward_type` | `wait-clip` | `queue` / `pressure` / `wait-clip` |
| `--algo` | `ddqn` | `dqn` / `ddqn` (DQN script only) |
| `--seed` | `42` | Random seed |
| `--total_steps` | `500000` | Training timesteps |
| `--save_dir` | — | Output directory for model checkpoints |
| `--gui` | off | Launch SUMO-GUI for visualisation |

---

## Evaluation

```bash
python evaluate_all.py
```

Or evaluate a specific scenario:

```bash
python evaluate_cologne8.py
python evaluate_cologne3_real.py
```

Evaluation is performed on the real TAPAS Cologne morning-peak traffic (07:00–08:00). Results are reported as mean ± std across 5 independent seeds.

---

## Algorithms

| Algorithm | Type | Key mechanism |
|---|---|---|
| **PPO** | On-policy actor-critic | Clipped surrogate objective, GAE |
| **DQN** | Off-policy value-based | Experience replay, target network |
| **DDQN** | Off-policy value-based | Decoupled action selection & evaluation |

All three are implemented with **parameter sharing**: every agent at every intersection shares a single neural network (MLP, 2 × 64, Tanh), while acting independently from local observations only.

---

## Reward functions

| Reward | Formula | Sensor requirement |
|---|---|---|
| **Queue** | Reduction in stopped vehicles + absolute queue penalty | Incoming lanes only |
| **Pressure** | Incoming − outgoing vehicle imbalance (clipped) | Incoming + outgoing lanes |
| **Wait-clip** | Cumulative waiting time of all vehicles (clipped) | Per-vehicle tracking |

---

## Scenarios

Both scenarios are derived from the [TAPAS Cologne dataset](https://sumo.dlr.de/docs/Data/Scenarios/TAPASCologne.html) and run at 30 % of full demand.

| Scenario | Intersections | Nodes | Edges | Vehicles |
|---|---|---|---|---|
| **Cologne-3** | 3 | 29 | 48 | 2,856 |
| **Cologne-8** | 8 | 78 | 149 | 2,046 |

---

## Citation

If you use this code, please cite the thesis:

```
Tran Linh Chi. Reinforcement Learning-based Traffic Signal Control Optimization in SUMO.
Bachelor's Thesis, University of Engineering and Technology,
Vietnam National University, Hanoi, 2026.
```
