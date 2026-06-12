import gymnasium as gym
import numpy as np


class SkipFrame(gym.Wrapper):
    """Return only every `skip`-th frame, summing rewards."""

    def __init__(self, env: gym.Env, skip: int):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
