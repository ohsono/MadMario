"""CPU-side replay buffer — uint8 frame storage, float32 tensors on sample.

Two memory optimizations vs the original tutorial:
1. Buffer lives in RAM as numpy (original stored CUDA tensors: ~22 GB VRAM
   at 100 K capacity); only the sampled batch moves to the device.
2. Observations are stored as uint8 (the [0,1] float32 frames are scaled by
   255) — 8x smaller, so a 100 K buffer of stacked 4x84x84 frame pairs fits
   in ~5.6 GB instead of ~45 GB. Quantization error is <=1/510 per pixel,
   which is exactly the precision the original uint8 NES frames had anyway.
"""
import random
from collections import deque
from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.memory: deque = deque(maxlen=capacity)

    @staticmethod
    def _encode(frames) -> np.ndarray:
        """[0,1] float frames -> uint8."""
        x = np.asarray(frames, dtype=np.float32).clip(0.0, 1.0)
        return (x * 255.0).round().astype(np.uint8)

    def push(
        self,
        state,
        next_state,
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        self.memory.append(
            (
                self._encode(state),
                self._encode(next_state),
                int(action),
                float(reward),
                bool(done),
            )
        )

    def sample(self, batch_size: int, device: torch.device) -> Tuple:
        batch = random.sample(self.memory, batch_size)
        state, next_state, action, reward, done = zip(*batch)
        return (
            torch.tensor(np.array(state), dtype=torch.float32, device=device) / 255.0,
            torch.tensor(np.array(next_state), dtype=torch.float32, device=device) / 255.0,
            torch.tensor(action, dtype=torch.long, device=device),
            torch.tensor(reward, dtype=torch.float32, device=device),
            torch.tensor(done, dtype=torch.bool, device=device),
        )

    def __len__(self) -> int:
        return len(self.memory)
