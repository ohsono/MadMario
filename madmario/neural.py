import copy

import torch
from torch import nn


class MarioNet(nn.Module):
    """Double DQN: online + frozen target networks share the same CNN architecture.

    Architecture: (Conv2d+ReLU)×3 → Flatten → (Linear+ReLU)×1 → Linear → Q-values
    Fixed 3136-unit linear layer assumes 84×84 input (enforced in __init__).
    """

    def __init__(self, input_dim: tuple, output_dim: int):
        super().__init__()
        c, h, w = input_dim
        if h != 84 or w != 84:
            raise ValueError(f"Expected 84×84 input, got {h}×{w}")

        self.online = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, model: str) -> torch.Tensor:
        if model == "online":
            return self.online(x)
        elif model == "target":
            return self.target(x)
        else:
            raise ValueError(f"model must be 'online' or 'target', got {model!r}")
