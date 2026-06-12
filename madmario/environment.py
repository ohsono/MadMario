"""Environment factory — builds the full preprocessing pipeline.

Migrated from gym → gymnasium (gym_super_mario_bros 9.x uses gymnasium internally).
The old gym.wrappers failed isinstance checks because observation spaces were
gymnasium.spaces.Box, not gym.spaces.Box.
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import gym_super_mario_bros
import numpy as np
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
    TransformObservation,
)
from nes_py.wrappers import JoypadSpace

from madmario.config import EnvConfig
from madmario.wrappers import SkipFrame


def make_env(config: EnvConfig, seed: Optional[int] = None) -> gym.Env:
    env = gym_super_mario_bros.make(config.env_name)
    env = JoypadSpace(env, config.action_space)
    env = SkipFrame(env, skip=config.skip_frames)
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=(84, 84))
    env = TransformObservation(
        env,
        func=lambda x: (x / 255.0).astype(np.float32),
        observation_space=gym.spaces.Box(
            low=0.0, high=1.0, shape=(84, 84), dtype=np.float32
        ),
    )
    env = FrameStackObservation(env, stack_size=config.stack_frames)

    if seed is not None:
        env.reset(seed=seed)

    return env
