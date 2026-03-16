import os
import sys
import argparse
import gymnasium as gym
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import csv
import numpy as np
import time

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from sumo_rl import SumoEnvironment

from stable_baselines3.common.callbacks import BaseCallback
import csv

class CsvLossCallback(BaseCallback):
    def __init__(self, log_file_path, verbose=0):
        super().__init__(verbose)
        self.log_file_path = log_file_path
        self.csv_file = None
        self.csv_writer = None

    def _on_training_start(self) -> None:
        self.csv_file = open(self.log_file_path, "w", newline="")
        fieldnames = [
            'timestep', 'loss', 'value_loss', 'entropy_loss',
            'approx_kl', 'clip_fraction', 'explained_variance' , 'policy_gradient_loss'
        ]
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()

    def _on_rollout_end(self) -> None:
        try:
            logs = self.model.logger.name_to_value
            row = {
                'timestep': self.num_timesteps,
                'loss': logs.get('train/loss', None),
                'value_loss': logs.get('train/value_loss', None),
                'entropy_loss': logs.get('train/entropy_loss', None),
                'approx_kl': logs.get('train/approx_kl', None),
                'clip_fraction': logs.get('train/clip_fraction', None),
                'explained_variance': logs.get('train/explained_variance', None),
                'policy_gradient_loss': logs.get('train/policy_gradient_loss', None)
            }
            self.csv_writer.writerow(row)
            self.csv_file.flush()
        except Exception as e:
            print(f"Logging failed on rollout end: {e}")


    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if self.csv_file:
            self.csv_file.close()

if __name__ == "__main__":

    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="PPO-Learning Single-Intersection with Enhanced Logging",
    )
    prs.add_argument("-gui", action="store_true", default=False)
    prs.add_argument("-r", dest="reward", type=str, default="queue", required=False)
    prs.add_argument(
        "-outputdir",
        dest="output_dir",
        type=str,
        default="/output",
    )
    prs.add_argument("-savefreq", dest="save_freq", type=int, default=10000)
    prs.add_argument("-trainsteps", dest="train_steps", type=int, default=100000)
    prs.add_argument("-testseconds", dest="test_seconds", type=int, default=1000)
    prs.add_argument("-seed", dest="seed", type=int, default=42, help="Random seed for reproducibility")

    args = prs.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_save_prefix = os.path.join(args.output_dir, "ppo_sumo_model")
    loss_log_file = os.path.join(args.output_dir, "training_loss_log.csv")
    checkpoint_save_path = os.path.join(args.output_dir, "checkpoints")
    train_csv_name = os.path.join(args.output_dir, "sumo_train_run.csv")
    test_csv_name = os.path.join(args.output_dir, "sumo_test_run.csv")
    test_system_log_file = os.path.join(args.output_dir, "test_system_log.csv")
    test_cycle_log_file = os.path.join(args.output_dir, "test_cycle_log.csv")

    print("Starting Training Phase...")
    train_env = SumoEnvironment(
        net_file="nets/RLQ/caliberated_net.xml",
        route_file="nets/RLQ/train_flows.xml",
        ##absolute file path required
        out_csv_name=train_csv_name,
        use_gui=args.gui,
        single_agent=True,
        num_seconds=args.train_steps + 5000,
        reward_fn=args.reward,
        sumo_seed=args.seed 
    )

    loss_callback = CsvLossCallback(log_file_path=loss_log_file)
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=checkpoint_save_path,
        name_prefix="ppo_sumo_chkpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=0.000003,
        n_steps=200,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=os.path.join(args.output_dir, "tensorboard_logs")
    )

    print(f"Training for {args.train_steps} timesteps...")
    model.learn(
        total_timesteps=args.train_steps,
        callback=[loss_callback, checkpoint_callback],
        log_interval=1
    )

    final_model_path = os.path.join(args.output_dir, "ppo_sumo_final_model.zip")
    print(f"Training complete. Saving final model to {final_model_path}...")
    model.save(final_model_path)
    print("Final model saved.")
    train_env.close()
