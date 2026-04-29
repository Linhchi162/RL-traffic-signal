"""
grid_env.py — MultiAgentVecEnv wrapper cho 2x2 grid.

Boc TrafficControlEnv(single_agent=False) thanh SB3-compatible VecEnv.
4 agent (moi junction 1 agent), cung chia se 1 SUMO simulation.
Dung cho IPPO voi shared policy weights.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from typing import List, Optional, Any


class MultiAgentVecEnv(VecEnv):
    """
    Boc TrafficControlEnv (single_agent=False) thanh VecEnv voi n_envs = so junction.

    Moi 'env' trong VecEnv tuong ung voi 1 junction (agent).
    Tat ca dung chung SUMO simulation.
    SB3 huan luyen 1 shared policy tren tat ca agents (IPPO w/ weight sharing).

    Args:
        make_env: callable tra ve TrafficControlEnv(single_agent=False, ...)
    """

    def __init__(self, make_env):
        self.env = make_env()
        self.signal_ids: List[str] = self.env.signal_ids
        n = len(self.signal_ids)

        obs_sp  = self.env._obs_space
        act_sp  = self.env._action_space

        super().__init__(num_envs=n, observation_space=obs_sp, action_space=act_sp)

        obs_shape = obs_sp.shape
        self._cached_obs  = {sid: np.zeros(obs_shape, dtype=np.float32)
                             for sid in self.signal_ids}
        self._pending_actions: Optional[dict] = None
        self._last_stacked  = np.zeros((n,) + obs_shape, dtype=np.float32)

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def reset(self):
        obs_dict, _ = self.env.reset()
        for sid, obs in obs_dict.items():
            self._cached_obs[sid] = np.array(obs, dtype=np.float32)
        self._last_stacked = self._stack()
        return self._last_stacked

    def step_async(self, actions: np.ndarray):
        self._pending_actions = {
            sid: int(a) for sid, a in zip(self.signal_ids, actions)
        }

    def step_wait(self):
        obs_dict, rew_dict, done_dict, _info = self.env.step(self._pending_actions)

        # Cap nhat cache voi obs moi (cac agent chua den luot giu nguyen)
        for sid, obs in obs_dict.items():
            self._cached_obs[sid] = np.array(obs, dtype=np.float32)

        obs   = self._stack()
        rews  = np.array([rew_dict.get(sid, 0.0) for sid in self.signal_ids],
                         dtype=np.float32)
        dones = np.array([done_dict["__all__"]] * self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]

        # Auto-reset khi episode ket thuc
        if done_dict["__all__"]:
            reset_dict, _ = self.env.reset()
            for sid, o in reset_dict.items():
                self._cached_obs[sid] = np.array(o, dtype=np.float32)
            obs = self._stack()

        self._last_stacked = obs
        return obs, rews, dones, infos

    def close(self):
        self.env.close()

    # ------------------------------------------------------------------
    # Bat buoc phai implement cac method abstract cua VecEnv
    # ------------------------------------------------------------------

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [None] * self.num_envs

    def get_attr(self, attr_name, indices=None):
        return [None] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        pass

    def seed(self, seed=None):
        return [None] * self.num_envs

    # ------------------------------------------------------------------

    def _stack(self) -> np.ndarray:
        return np.stack([self._cached_obs[sid] for sid in self.signal_ids])
