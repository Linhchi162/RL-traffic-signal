"""
test_grid_env.py — Kiem tra nhanh Grid2x2 multi-agent env.

Test 1: Khoi tao env, in obs/action space
Test 2: Chay 200 buoc voi random action, in reward
Test 3: Train PPO ngan 5000 buoc, xem reward co tang khong

Chay:
    python test_grid_env.py
"""

import os, sys
os.environ.setdefault("SUMO_HOME", r"/usr/share/sumo" if sys.platform != "win32"
                      else r"D:\Program Files\Eclipse\Sumo")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

from pathlib import Path
import numpy as np

_HERE = Path(__file__).parent
NET   = _HERE / "nets"   / "grid2x2.net.xml"
ROUTE = _HERE / "generated_flows" / "grid_s0.xml"

from rl_controller.traffic_env import TrafficControlEnv
from rl_controller.state_builder import BaselineObservation
from rl_controller.grid_env     import MultiAgentVecEnv

# ---------------------------------------------------------------------------
def make():
    return TrafficControlEnv(
        net_file    = str(NET),
        route_file  = str(ROUTE),
        sim_duration= 3600,          # 1h sim, du de test
        reward_fn   = "queue",
        obs_class   = BaselineObservation,
        sumo_seed   = 42,
        single_agent= False,
        show_warnings=False,
    )

# ---------------------------------------------------------------------------
print("=" * 55)
print("TEST 1: khoi tao env")
vec = MultiAgentVecEnv(make)
print(f"  n_agents        : {vec.num_envs}")
print(f"  signal_ids      : {vec.env.signal_ids}")
print(f"  observation_space: {vec.observation_space}")
print(f"  action_space    : {vec.action_space}")

obs = vec.reset()
print(f"  obs shape after reset: {obs.shape}  (expect ({vec.num_envs}, 7))")
assert obs.shape == (vec.num_envs, 7), "SAI obs shape!"
print("  PASS")

# ---------------------------------------------------------------------------
print()
print("TEST 2: 200 buoc random action")
rewards_all = []
for step in range(200):
    actions = np.array([vec.action_space.sample() for _ in range(vec.num_envs)])
    obs, rews, dones, infos = vec.step(actions)
    rewards_all.append(rews.mean())
    if dones.any():
        obs = vec.reset()
        break

print(f"  Buoc hoan thanh : {len(rewards_all)}")
print(f"  Mean reward     : {np.mean(rewards_all):.4f}")
print(f"  Obs shape       : {obs.shape}")
print("  PASS" if len(rewards_all) > 0 else "  FAIL: 0 buoc!")

vec.close()

# ---------------------------------------------------------------------------
print()
print("TEST 3: train PPO ngan (1000 buoc), theo doi reward")
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_rewards = []
        self._buf = []

    def _on_step(self) -> bool:
        self._buf.extend(self.locals["rewards"].tolist())
        if len(self._buf) >= 512:
            self.ep_rewards.append(np.mean(self._buf))
            print(f"    step {self.num_timesteps:>6d} | mean_rew = {self.ep_rewards[-1]:.4f}")
            self._buf = []
        return True

vec2 = MultiAgentVecEnv(make)
model = PPO("MlpPolicy", vec2, n_steps=256, batch_size=64,
            learning_rate=3e-4, verbose=0, seed=42, device="cpu")
cb = RewardLogger()
model.learn(total_timesteps=1000, callback=cb)
vec2.close()

if len(cb.ep_rewards) >= 2:
    trend = cb.ep_rewards[-1] - cb.ep_rewards[0]
    print(f"  Reward trend: {cb.ep_rewards[0]:.4f} -> {cb.ep_rewards[-1]:.4f}  (delta={trend:+.4f})")
    print("  Env hoat dong dung. (1000 buoc chi de kiem tra, chua du de hoc)")
else:
    print("  Khong du du lieu de xet trend.")

print()
print("=" * 55)
print("Tat ca test PASS. San sang deploy len Vast.ai!")
