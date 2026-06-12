import numpy as np
import pytest
import gymnasium as gym
from wrappers import SkipFrame


class _FakeEnv(gym.Env):
    """Minimal deterministic env for wrapper testing."""

    observation_space = gym.spaces.Box(0, 255, shape=(8, 8, 3), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, ep_len: int = 10):
        super().__init__()
        self._t = 0
        self._ep_len = ep_len

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return np.zeros((8, 8, 3), dtype=np.uint8), {}

    def step(self, action):
        self._t += 1
        obs = np.full((8, 8, 3), self._t, dtype=np.uint8)
        reward = float(self._t)
        terminated = self._t >= self._ep_len
        return obs, reward, terminated, False, {"t": self._t}


def test_skipframe_reward_sum():
    """SkipFrame must accumulate rewards over `skip` steps."""
    env = SkipFrame(_FakeEnv(ep_len=20), skip=4)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    # _FakeEnv returns reward=t; with skip=4, t goes 1→4, sum=1+2+3+4=10
    assert reward == 10.0


def test_skipframe_early_termination():
    """SkipFrame must stop accumulating once terminated."""
    env = SkipFrame(_FakeEnv(ep_len=2), skip=4)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated is True
    # Only 2 steps ran: reward = 1+2 = 3
    assert reward == 3.0


def test_skipframe_obs_is_last_frame():
    """SkipFrame must return the observation from the final step."""
    env = SkipFrame(_FakeEnv(ep_len=20), skip=4)
    env.reset()
    obs, *_ = env.step(0)
    # After 4 steps (t=1..4), last obs fills with t=4
    assert obs[0, 0, 0] == 4
