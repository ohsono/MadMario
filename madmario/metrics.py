"""Metric logger with W&B integration.

Bug fix: log_step used `if loss:` which silently dropped loss=0.0 as falsy.
Fixed to `if loss is not None:`.
"""
from __future__ import annotations

import datetime
import time
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import wandb as _wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class MetricLogger:
    def __init__(
        self,
        save_dir: Path,
        use_wandb: bool = False,
        **wandb_init_kwargs: Any,
    ):
        self.save_dir = save_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )

        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        self.ep_rewards: list = []
        self.ep_lengths: list = []
        self.ep_avg_losses: list = []
        self.ep_avg_qs: list = []

        self.moving_avg_ep_rewards: list = []
        self.moving_avg_ep_lengths: list = []
        self.moving_avg_ep_avg_losses: list = []
        self.moving_avg_ep_avg_qs: list = []

        self.init_episode()
        self.record_time = time.time()

        if self.use_wandb:
            _wandb.init(**wandb_init_kwargs)

    # ------------------------------------------------------------------

    def log_step(
        self,
        reward: float,
        loss: Optional[float],
        q: Optional[float],
    ) -> None:
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        # FIX: was `if loss:` — drops loss=0.0 silently; must be `is not None`
        if loss is not None:
            self.curr_ep_loss += loss
            self.curr_ep_q += q  # type: ignore[operator]
            self.curr_ep_loss_length += 1

    def log_episode(self) -> None:
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss, ep_avg_q = 0.0, 0.0
        else:
            ep_avg_loss = round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        if self.use_wandb:
            _wandb.log(
                {
                    "ep_reward": self.curr_ep_reward,
                    "ep_length": self.curr_ep_length,
                    "ep_avg_loss": ep_avg_loss,
                    "ep_avg_q": ep_avg_q,
                    "episode": len(self.ep_rewards),
                }
            )

        self.init_episode()

    def init_episode(self) -> None:
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode: int, epsilon: float, step: int) -> Dict[str, float]:
        mean_ep_reward = round(float(np.mean(self.ep_rewards[-100:])), 3)
        mean_ep_length = round(float(np.mean(self.ep_lengths[-100:])), 3)
        mean_ep_loss = round(float(np.mean(self.ep_avg_losses[-100:])), 3)
        mean_ep_q = round(float(np.mean(self.ep_avg_qs[-100:])), 3)

        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last = self.record_time
        self.record_time = time.time()
        delta = round(self.record_time - last, 3)

        print(
            f"Episode {episode} | Step {step} | ε {epsilon:.4f} | "
            f"Reward {mean_ep_reward} | Length {mean_ep_length} | "
            f"Loss {mean_ep_loss} | Q {mean_ep_q} | Δt {delta}s"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}"
                f"{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{delta:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        metrics: Dict[str, float] = {
            "moving_avg_reward": mean_ep_reward,
            "moving_avg_length": mean_ep_length,
            "moving_avg_loss": mean_ep_loss,
            "moving_avg_q": mean_ep_q,
            "epsilon": epsilon,
            "step": step,
            "episode": episode,
        }
        if self.use_wandb:
            _wandb.log(metrics)

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()

        return metrics

    def close(self) -> None:
        if self.use_wandb:
            _wandb.finish()
