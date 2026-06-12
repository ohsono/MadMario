"""CPU-side replay buffer — stores raw numpy arrays, converts to tensors on sample.

Bug fix: original code stored CUDA tensors in the deque, requiring ~22 GB VRAM
for a 100K buffer.  Storing float32 numpy arrays keeps the buffer in RAM (~1.8 GB)
and only moves a 32-sample batch to the device during each learn step.
"""
import random
from collections import deque
from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.memory: deque = deque(maxlen=capacity)

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
                np.array(state, dtype=np.float32),
                np.array(next_state, dtype=np.float32),
                int(action),
                float(reward),
                bool(done),
            )
        )

    def sample(self, batch_size: int, device: torch.device) -> Tuple:
        batch = random.sample(self.memory, batch_size)
        state, next_state, action, reward, done = zip(*batch)
        return (
            torch.tensor(np.array(state), dtype=torch.float32, device=device),
            torch.tensor(np.array(next_state), dtype=torch.float32, device=device),
            torch.tensor(action, dtype=torch.long, device=device),
            torch.tensor(reward, dtype=torch.float32, device=device),
            torch.tensor(done, dtype=torch.bool, device=device),
        )

    def __len__(self) -> int:
        return len(self.memory)
