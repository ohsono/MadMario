"""Population-based multi-agent training.

Each agent runs in its own process with its own environment and replay buffer.
The coordinator collects per-episode results, identifies the best-performing agent,
and periodically broadcasts its network weights to the rest of the population.

Usage (via train.py --multi-agent):
    python train.py train --multi-agent --num-agents 4 --episodes 5000
"""
from __future__ import annotations

import multiprocessing as mp
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from config import Config, MultiAgentConfig


# ---------------------------------------------------------------------------
# Worker — runs in a subprocess
# ---------------------------------------------------------------------------

def _agent_worker(
    agent_id: int,
    cfg: Config,
    save_dir: Path,
    episodes: int,
    result_queue: mp.Queue,
    weight_queue: mp.Queue,
    seed: int,
) -> None:
    """Single-agent training process. Sends episode results back to coordinator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    from environment import make_env
    from agent import Mario

    agent_save_dir = save_dir / f"agent_{agent_id}"
    agent_save_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(cfg.env, seed=seed)
    mario = Mario(
        state_dim=cfg.state_dim,
        action_dim=env.action_space.n,
        save_dir=agent_save_dir,
        config=cfg.agent,
    )

    for episode in range(episodes):
        # Non-blocking check for updated weights from coordinator
        if not weight_queue.empty():
            try:
                weights = weight_queue.get_nowait()
                mario.net.load_state_dict(weights)
            except Exception:
                pass

        obs, _ = env.reset()
        total_reward, flag_get = 0.0, False
        while True:
            action = mario.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            mario.cache(obs, next_obs, action, reward, done)
            mario.learn()
            total_reward += reward
            obs = next_obs
            if done or info.get("flag_get"):
                flag_get = bool(info.get("flag_get"))
                break

        payload: Dict[str, Any] = {
            "agent_id": agent_id,
            "episode": episode,
            "reward": total_reward,
            "flag_get": flag_get,
            "step": mario.curr_step,
        }
        # Every sync_interval episodes, include weights for coordinator evaluation
        if episode % cfg.multi_agent.sync_interval == 0:
            payload["weights"] = {
                k: v.cpu() for k, v in mario.net.state_dict().items()
            }
        result_queue.put(payload)

    env.close()


# ---------------------------------------------------------------------------
# Coordinator — runs in the main process
# ---------------------------------------------------------------------------

class MultiAgentCoordinator:
    """Launches N agent workers and performs population-level best-weight sync."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mac: MultiAgentConfig = cfg.multi_agent

    def run(self, episodes: int, save_dir: Path) -> Dict[int, List[float]]:
        n = self.mac.num_agents
        save_dir.mkdir(parents=True, exist_ok=True)

        result_queue: mp.Queue = mp.Queue()
        weight_queues: List[mp.Queue] = [mp.Queue() for _ in range(n)]

        processes = [
            mp.Process(
                target=_agent_worker,
                args=(
                    i,
                    self.cfg,
                    save_dir,
                    episodes,
                    result_queue,
                    weight_queues[i],
                    42 + i,
                ),
                daemon=True,
            )
            for i in range(n)
        ]
        for p in processes:
            p.start()
        print(f"[MultiAgent] {n} agents × {episodes} episodes started.")

        agent_rewards: Dict[int, List[float]] = {i: [] for i in range(n)}
        best_agent_id: Optional[int] = None
        best_mean: float = -float("inf")
        total = 0
        expected = n * episodes

        while total < expected:
            # Check if all workers died unexpectedly
            if not any(p.is_alive() for p in processes):
                print("[MultiAgent] All workers exited early.")
                break

            try:
                result = result_queue.get(timeout=60)
            except Exception:
                continue

            aid = result["agent_id"]
            agent_rewards[aid].append(result["reward"])
            total += 1

            # Evaluate and potentially broadcast best weights
            if "weights" in result:
                recent_mean = float(np.mean(agent_rewards[aid][-20:]))
                if recent_mean > best_mean:
                    best_mean = recent_mean
                    best_agent_id = aid
                    for j, q in enumerate(weight_queues):
                        if j != aid:
                            q.put(result["weights"])
                    print(
                        f"[MultiAgent] New best: agent {aid} "
                        f"(20-ep mean {best_mean:.1f}) → broadcasting weights"
                    )

            if total % (n * 20) == 0:
                means = {
                    i: round(float(np.mean(rs[-20:])), 1) if rs else 0.0
                    for i, rs in agent_rewards.items()
                }
                print(f"[MultiAgent] {total}/{expected} | 20-ep means: {means}")

        for p in processes:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()

        print(
            f"[MultiAgent] Done. Best agent: {best_agent_id} "
            f"(mean {best_mean:.1f})"
        )
        return agent_rewards
