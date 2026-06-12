"""Mario DQN agent — Double DQN with fixed replay buffer and checkpoint handling.

Bug fixes applied:
1. GPU OOM: replay buffer stores numpy arrays on CPU; tensors created per-batch.
2. curr_step persisted in checkpoint so resumed training continues correctly.
3. Optimizer state saved/restored so Adam moments survive restarts.
4. Burnin guard is checked FIRST in learn(); sync/save only happen after burnin.
5. reward stored as float32 (was float64), avoiding spurious dtype upcast in td_target.
6. td_estimate/td_target use torch.arange tied to actual batch size, not a hardcoded constant.
7. MarioNet.forward raises ValueError on invalid model string (see neural.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from config import AgentConfig
from neural import MarioNet
from replay import ReplayBuffer


class Mario:
    def __init__(
        self,
        state_dim: tuple,
        action_dim: int,
        save_dir: Path,
        config: AgentConfig,
        checkpoint: Optional[Path] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.cfg = config

        self.curr_step = 0
        self.learn_step_count = 0  # gradient steps taken (used by train_step)
        self.exploration_rate = config.exploration_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = MarioNet(state_dim, action_dim).float().to(self.device)
        # FIX 1: CPU-side buffer; tensors only move to device during sample()
        self.memory = ReplayBuffer(config.memory_size)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr)
        self.loss_fn = torch.nn.SmoothL1Loss()

        if checkpoint:
            self.load(checkpoint)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def act(self, state) -> int:
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            s = torch.tensor(
                np.array(state), dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            action_idx = torch.argmax(self.net(s, model="online"), dim=1).item()

        self.exploration_rate = max(
            self.cfg.exploration_rate_min,
            self.exploration_rate * self.cfg.exploration_rate_decay,
        )
        self.curr_step += 1
        return int(action_idx)

    def cache(self, state, next_state, action: int, reward: float, done: bool) -> None:
        # FIX 1: delegate to CPU buffer (no .cuda() here)
        self.memory.push(state, next_state, action, reward, done)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def recall(self):
        return self.memory.sample(self.cfg.batch_size, self.device)

    def td_estimate(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # FIX 6: index using actual batch dimension, not hardcoded self.batch_size
        idx = torch.arange(state.shape[0], device=self.device)
        return self.net(state, model="online")[idx, action]

    @torch.no_grad()
    def td_target(
        self,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        best_action = torch.argmax(self.net(next_state, model="online"), dim=1)
        idx = torch.arange(next_state.shape[0], device=self.device)
        next_Q = self.net(next_state, model="target")[idx, best_action]
        # FIX 5: reward is float32 from ReplayBuffer; no implicit float64 upcast
        return reward + (~done) * self.cfg.gamma * next_Q

    def update_Q_online(
        self, td_estimate: torch.Tensor, td_target: torch.Tensor
    ) -> float:
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self) -> None:
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self) -> Tuple[Optional[float], Optional[float]]:
        # FIX 4: burnin guard FIRST — sync/save only run after burnin
        if self.curr_step < self.cfg.burnin:
            return None, None
        if self.curr_step % self.cfg.learn_every != 0:
            return None, None
        # Buffer must hold at least one batch regardless of burnin setting
        if len(self.memory) < self.cfg.batch_size:
            return None, None

        if self.curr_step % self.cfg.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step % self.cfg.save_every == 0:
            self.save()

        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)
        return td_est.mean().item(), loss

    def train_step(self) -> Tuple[Optional[float], Optional[float]]:
        """One unconditional gradient step from the buffer (for external learners,
        e.g. the shared-experience multi-agent coordinator, which drives cadence
        itself). Target sync / checkpointing are keyed to gradient steps at the
        same effective rate as learn(): sync_every/learn_every env-steps."""
        if len(self.memory) < self.cfg.batch_size:
            return None, None

        self.learn_step_count += 1
        sync_period = max(1, self.cfg.sync_every // self.cfg.learn_every)
        save_period = max(1, self.cfg.save_every // self.cfg.learn_every)
        if self.learn_step_count % sync_period == 0:
            self.sync_Q_target()
        if self.learn_step_count % save_period == 0:
            self.save()

        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)
        return td_est.mean().item(), loss

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> Path:
        slot = int(self.curr_step // self.cfg.save_every)
        save_path = self.save_dir / f"mario_net_{slot}.chkpt"
        torch.save(
            {
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),  # FIX 3
                "exploration_rate": self.exploration_rate,
                "curr_step": self.curr_step,               # FIX 2
            },
            save_path,
        )
        print(f"Saved → {save_path} (step {self.curr_step})")
        return save_path

    def load(self, load_path: Path) -> None:
        if not Path(load_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")
        ckp = torch.load(load_path, map_location=self.device, weights_only=False)
        if ckp.get("model") is None:
            raise KeyError(f"Checkpoint at {load_path} missing 'model' key")
        self.net.load_state_dict(ckp["model"])
        if "optimizer" in ckp:                             # FIX 3
            self.optimizer.load_state_dict(ckp["optimizer"])
        self.exploration_rate = ckp.get("exploration_rate", self.exploration_rate)
        self.curr_step = ckp.get("curr_step", 0)          # FIX 2
        print(
            f"Loaded {load_path} "
            f"(step {self.curr_step}, ε={self.exploration_rate:.4f})"
        )
