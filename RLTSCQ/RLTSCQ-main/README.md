# Reinforcement Learning Based Traffic Signal Design to Minimize Queue Lengths
This is the official codebase for our paper on arXiv: [arXiv:2509.21745](https://arxiv.org/abs/2509.21745)

This repository contains the code and experiments for our paper on designing traffic signal controllers using Reinforcement Learning (RL).  The framework utilizes the [SUMO (Simulation of Urban MObility)](https://github.com/eclipse/sumo) traffic simulator.

## Project Structure

<details>
<summary>Click to view the project directory tree</summary>

```
.
├── experiments
│   ├── ppo_test.py
│   └── ppo_train.py
├── setup.py
├── sumo_rl
│   ├── agents
│   ├── environment
│   ├── exploration
│   ├── nets
│   └── util
└── tests
|   ├── gym_test.py
|   └── pz_test.py
|
|── requirements.txt
|── README.md

```

</details>

## Installation

### 1. Install SUMO

First, you need to install the SUMO simulator. The following instructions are for Ubuntu/Debian-based systems.

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

Next, you must set the `SUMO_HOME` environment variable. The default installation path is `/usr/share/sumo`.

```bash
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```

**Important**: For a significant performance boost (~8x) when running without the GUI, you can use Libsumo. Set the following environment variable:

```bash
export LIBSUMO_AS_TRACI=1
```

Note that with this variable enabled, you cannot run `sumo-gui` or parallel simulations. See the [Libsumo documentation](https://sumo.dlr.de/docs/Libsumo.html) for more details.

### 2. Install Python Dependencies

This project contains a `requirements.txt` file with all the necessary Python packages.

First, clone the repository:

```bash
git clone https://github.com/AnirudN/RLTSCQ.git
cd <RLTSCQ>
```

It is highly recommended to use a virtual environment (like `conda` or `venv`) to manage dependencies.

```bash
# Create and activate a conda environment
conda create -n tsc_rl python=3.11.11
conda activate tsc_rl

# Install dependencies from the requirements file
pip install -r requirements.txt
```

For development purposes, you can install the project in editable mode:

```bash
pip install -e .
```

## How to Run Experiments

This project uses a PPO (Proximal Policy Optimization) agent for training and testing.

### Configuration

Before running the experiments, you may need to configure the environment:

- **State Representation (Observation Space)**: To change the observation class, navigate to `RLTSCQ/sumo_rl/environment/env.py` and modify line 104.

  **File**: `RLTSCQ/sumo_rl/environment/env.py`

  **Line to change**: `observation_class: ObservationFunction = AutoTrainingProjector_Latent_16`

  **Available Classes** (defined in `RLTSCQ/sumo_rl/environment/observations.py`):
  - `AutoTrainingProjector_Latent_4`
  - `AutoTrainingProjector_Latent_8`
  - `AutoTrainingProjector_Latent_16`
  - `AutoTrainingProjector_Latent_19`
  - `AutoTrainingProjector_Latent_32`
  - `KPlanesStateRepresentation`
  - `FullStateObservation` (19D state)
  - `MyObservationWithDurationsDirectTraci` (Baseline)

- **Autoencoder Training File Path**: If using an autoencoder-based state representation, you must set your desired save directory for the AE training files.

  **File**: `RLTSCQ/sumo_rl/environment/observations.py`

  **Line to change**: `self.save_dir = "/seeding_results/autoencoder_dimension_comparison"` (around line 676). Please use an absolute path.

- **Network and Route Files**: The environment definition in `RLTSCQ/sumo_rl/environment/env.py` (lines 97-98) uses hardcoded paths for the SUMO network and traffic flows. Ensure these paths are correct for your setup.

  **Network File**: `nets/RLQ/caliberated_net.xml`

  **Route File**: `nets/RLQ/train_flows.xml`

### Training

To train a PPO agent, run the `ppo_train.py` script from the root directory.

```bash
python RLTSCQ/experiments/ppo_train.py --outputdir <path_to_output_directory> --reward <reward_function>
```

**Arguments**:
- `--outputdir`: (Required) Path to the directory where training logs and the final model will be saved.
- `--reward`: (Optional) The reward function to use. Available options: `queue`, `diff-waiting-time`, `average-speed`. Default behavior uses the reward function specified in the environment file.

### Testing

After training, you can evaluate the agent's performance using the `ppo_test.py` script.

```bash
python RLTSCQ/experiments/ppo_test.py --outputdir <path_to_output_directory> --model_path <path_to_model.zip>
```

**Arguments**:
- `--outputdir`: (Required) Path to the directory where test results (e.g., CSV files) will be saved.
- `--model_path`: (Required) Path to the trained model file. This is typically found in your training output directory as `ppo_sumo_final_model.zip`.

**Note**: The reward function for testing is set to `queue` by default. To change this, you can modify the `test_env` variable definition in `RLTSCQ/experiments/ppo_test.py` (around line 136).

## Citing This Work

If you use this code in your research, please cite our paper:

[Link to Paper](https://arxiv.org/abs/2509.21745) <!-- Replace with your actual paper link -->

```bibtex
@article{nandakumar2025reinforcement,
  title={Reinforcement Learning Based Traffic Signal Design to Minimize Queue Lengths},
  author={Nandakumar, Anirud and Banerjee, Chayan and Vanajakshi, Lelitha Devi},
  journal={arXiv preprint arXiv:2509.21745},
  year={2025}
}
```

## Acknowledgments

This project was built on top of the `sumo-rl` library. We extend our gratitude to its creators for providing a robust foundation for reinforcement learning research in traffic simulation.

```bibtex
@misc{AlegreSUMORL,
    author = {Lucas N. Alegre},
    title = {{SUMO-RL}},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/LucasAlegre/sumo-rl}},
}
```
