from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class EnvConfig:
    world: int = 1
    stage: int = 1
    version: int = 0
    action_space: List[List[str]] = field(
        default_factory=lambda: [["right"], ["right", "A"]]
    )
    skip_frames: int = 4
    stack_frames: int = 4

    @property
    def env_name(self) -> str:
        return f"SuperMarioBros-{self.world}-{self.stage}-v{self.version}"


@dataclass
class AgentConfig:
    batch_size: int = 32
    exploration_rate: float = 1.0
    exploration_rate_decay: float = 0.99999975
    exploration_rate_min: float = 0.1
    gamma: float = 0.9
    lr: float = 0.00025
    burnin: int = 100_000
    learn_every: int = 3
    sync_every: int = 10_000
    save_every: int = 500_000
    memory_size: int = 100_000


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "madmario"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    log_video_every: int = 500


@dataclass
class TrainConfig:
    episodes: int = 40_000
    seed: int = 42
    checkpoint: Optional[Path] = None
    save_dir: Optional[Path] = None

    def resolve_save_dir(self) -> Path:
        if self.save_dir:
            return Path(self.save_dir)
        return Path("checkpoints") / datetime.datetime.now().strftime(
            "%Y-%m-%dT%H-%M-%S"
        )


@dataclass
class AutonomousConfig:
    enabled: bool = False
    curriculum_enabled: bool = True
    curriculum_advance_threshold: float = 0.70
    curriculum_window: int = 100
    plateau_patience: int = 200
    plateau_min_delta: float = 0.5


@dataclass
class MultiAgentConfig:
    enabled: bool = False
    num_agents: int = 4
    mode: str = "shared"  # "shared" (Ape-X actor-learner) | "pbt" (population)
    # shared mode: actor i uses fixed eps = eps_base * eps_alpha**(i/(N-1))
    eps_base: float = 0.4
    eps_alpha: float = 1 / 128
    weight_sync_steps: int = 200  # gradient steps between weight broadcasts
    # pbt mode
    pbt_interval: int = 25     # episodes between PBT exploit/explore events
    pbt_quantile: float = 0.25  # truncation-selection quantile


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    autonomous: AutonomousConfig = field(default_factory=AutonomousConfig)
    multi_agent: MultiAgentConfig = field(default_factory=MultiAgentConfig)
    state_dim: tuple = (4, 84, 84)
