"""Autonomous RL — curriculum learning and self-improvement.

CurriculumManager   — tracks per-level success rate, auto-advances world/stage
PlateauDetector     — detects reward stagnation for hyperparameter adaptation
SelfImprovementLoop — orchestrates curriculum + plateau response during training
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from madmario.config import AutonomousConfig, EnvConfig

# Ordered list of (world, stage) pairs for curriculum progression
CURRICULUM_LEVELS: List[Tuple[int, int]] = [
    (1, 1), (1, 2), (1, 3), (1, 4),
    (2, 1), (2, 2), (2, 3), (2, 4),
    (3, 1), (3, 2), (3, 3), (3, 4),
    (4, 1), (4, 2), (4, 3), (4, 4),
    (5, 1), (5, 2), (5, 3), (5, 4),
    (6, 1), (6, 2), (6, 3), (6, 4),
    (7, 1), (7, 2), (7, 3), (7, 4),
    (8, 1), (8, 2), (8, 3), (8, 4),
]


class CurriculumManager:
    """Tracks Mario's completion rate and advances to the next level automatically."""

    def __init__(self, config: AutonomousConfig, start_world: int = 1, start_stage: int = 1):
        self.cfg = config
        try:
            self._level_idx = CURRICULUM_LEVELS.index((start_world, start_stage))
        except ValueError:
            self._level_idx = 0
        self._recent_flags: deque = deque(maxlen=config.curriculum_window)

    @property
    def current_level(self) -> Tuple[int, int]:
        return CURRICULUM_LEVELS[self._level_idx]

    @property
    def success_rate(self) -> float:
        if not self._recent_flags:
            return 0.0
        return float(np.mean(list(self._recent_flags)))

    def record_episode(self, flag_get: bool) -> None:
        self._recent_flags.append(float(flag_get))

    def should_advance(self) -> bool:
        return (
            len(self._recent_flags) >= self.cfg.curriculum_window
            and self.success_rate >= self.cfg.curriculum_advance_threshold
        )

    def advance(self) -> Optional[Tuple[int, int]]:
        if self._level_idx < len(CURRICULUM_LEVELS) - 1:
            self._level_idx += 1
            self._recent_flags.clear()
            return CURRICULUM_LEVELS[self._level_idx]
        return None  # already at final level

    def state_dict(self) -> Dict[str, Any]:
        return {
            "level_idx": self._level_idx,
            "recent_flags": list(self._recent_flags),
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self._level_idx = d["level_idx"]
        self._recent_flags = deque(d["recent_flags"], maxlen=self.cfg.curriculum_window)


class PlateauDetector:
    """Detects stagnation in a reward signal over a rolling window."""

    def __init__(self, patience: int, min_delta: float = 0.5):
        self.patience = patience
        self.min_delta = min_delta
        self._best: float = -float("inf")
        self._steps_without_improvement = 0

    def update(self, value: float) -> bool:
        """Returns True when a plateau is detected."""
        if value > self._best + self.min_delta:
            self._best = value
            self._steps_without_improvement = 0
            return False
        self._steps_without_improvement += 1
        return self._steps_without_improvement >= self.patience

    def reset(self) -> None:
        self._best = -float("inf")
        self._steps_without_improvement = 0


class SelfImprovementLoop:
    """High-level coordinator: wraps a Mario agent with curriculum + plateau responses.

    Usage:
        loop = SelfImprovementLoop(autonomous_cfg, env_cfg)
        # inside training loop:
        events = loop.on_episode_end(episode, reward, flag_get, mario_agent)
        if "curriculum_advance" in events:
            world, stage = events["curriculum_advance"]
            env.close(); env = make_env(updated_cfg)
    """

    def __init__(self, config: AutonomousConfig, env_config: EnvConfig):
        self.cfg = config
        self.curriculum = CurriculumManager(
            config, env_config.world, env_config.stage
        )
        self.plateau = PlateauDetector(config.plateau_patience, config.plateau_min_delta)
        self._best_reward: float = -float("inf")
        self._best_checkpoint: Optional[Path] = None

    def on_episode_end(
        self,
        episode: int,
        reward: float,
        flag_get: bool,
        agent: Any,
    ) -> Dict[str, Any]:
        """Call at end of each episode. Returns a dict of triggered events."""
        events: Dict[str, Any] = {}

        # --- curriculum ---
        self.curriculum.record_episode(flag_get)
        if self.curriculum.should_advance():
            new_level = self.curriculum.advance()
            if new_level:
                events["curriculum_advance"] = new_level
                print(
                    f"[Curriculum] Advancing to World {new_level[0]}-{new_level[1]} "
                    f"(success rate {self.curriculum.success_rate:.1%})"
                )

        # --- plateau response ---
        if self.plateau.update(reward):
            events["plateau"] = True
            old_min = agent.cfg.exploration_rate_min
            agent.cfg.exploration_rate_min = max(0.05, old_min * 0.9)
            print(
                f"[Plateau] {self.cfg.plateau_patience} episodes without improvement. "
                f"Lowering ε_min {old_min:.3f} → {agent.cfg.exploration_rate_min:.3f}"
            )
            self.plateau.reset()

        # --- best model tracking ---
        if reward > self._best_reward:
            self._best_reward = reward
            events["new_best"] = reward

        events["curriculum_level"] = self.curriculum.current_level
        events["curriculum_success_rate"] = self.curriculum.success_rate
        return events

    def state_dict(self) -> Dict[str, Any]:
        return {
            "curriculum": self.curriculum.state_dict(),
            "best_reward": self._best_reward,
        }
