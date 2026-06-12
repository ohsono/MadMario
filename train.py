#!/usr/bin/env python3
"""MadMario training CLI — replaces main.py.

Examples:
  python train.py train                          # default 40k episodes
  python train.py train --episodes 1000 --seed 7
  python train.py train --use-wandb --wandb-project mario-exp
  python train.py train --autonomous             # curriculum learning
  python train.py train --multi-agent --num-agents 4
  python train.py train --checkpoint checkpoints/.../mario_net_1.chkpt
  python train.py evaluate checkpoints/.../mario_net_1.chkpt
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer

app = typer.Typer(help="MadMario DQN trainer", add_completion=False)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# train command
# ---------------------------------------------------------------------------

@app.command()
def train(
    episodes: int = typer.Option(40_000, help="Training episodes"),
    checkpoint: Optional[Path] = typer.Option(None, help="Resume from checkpoint"),
    save_dir: Optional[Path] = typer.Option(None, help="Override checkpoint save dir"),
    seed: int = typer.Option(42),
    world: int = typer.Option(1, help="Mario world (1-8)"),
    stage: int = typer.Option(1, help="Mario stage (1-4)"),
    use_wandb: bool = typer.Option(False, help="Enable W&B logging"),
    wandb_project: str = typer.Option("madmario"),
    wandb_entity: Optional[str] = typer.Option(None),
    autonomous: bool = typer.Option(False, help="Curriculum + self-improvement"),
    multi_agent: bool = typer.Option(False, help="Parallel population training"),
    num_agents: int = typer.Option(4, help="Agents for --multi-agent"),
    burnin: int = typer.Option(100_000, help="Steps before training starts"),
    memory_size: int = typer.Option(100_000, help="Replay buffer capacity"),
) -> None:
    from config import (
        AgentConfig, AutonomousConfig, Config, EnvConfig,
        MultiAgentConfig, TrainConfig, WandbConfig,
    )

    cfg = Config(
        env=EnvConfig(world=world, stage=stage),
        agent=AgentConfig(burnin=burnin, memory_size=memory_size),
        wandb=WandbConfig(enabled=use_wandb, project=wandb_project, entity=wandb_entity),
        train=TrainConfig(
            episodes=episodes, seed=seed,
            checkpoint=checkpoint, save_dir=save_dir,
        ),
        autonomous=AutonomousConfig(enabled=autonomous),
        multi_agent=MultiAgentConfig(enabled=multi_agent, num_agents=num_agents),
    )
    _run(cfg)


# ---------------------------------------------------------------------------
# evaluate command
# ---------------------------------------------------------------------------

@app.command()
def evaluate(
    checkpoint_path: Path = typer.Argument(..., help="Path to .chkpt file"),
    n_episodes: int = typer.Option(10, help="Evaluation episodes"),
    world: int = typer.Option(1),
    stage: int = typer.Option(1),
    seed: int = typer.Option(0),
) -> None:
    from config import Config, EnvConfig
    from environment import make_env
    from agent import Mario
    import numpy as np

    _seed(seed)
    cfg = Config(env=EnvConfig(world=world, stage=stage))
    env = make_env(cfg.env, seed=seed)
    mario = Mario(
        state_dim=cfg.state_dim,
        action_dim=env.action_space.n,
        save_dir=checkpoint_path.parent,
        config=cfg.agent,
        checkpoint=checkpoint_path,
    )
    mario.exploration_rate = 0.0

    rewards, flags = [], 0
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        info: dict = {}
        while not done:
            action = mario.act(obs)
            obs, rew, term, trunc, info = env.step(action)
            total += rew
            done = term or trunc or bool(info.get("flag_get"))
        rewards.append(total)
        if info.get("flag_get"):
            flags += 1
        print(f"  ep {ep+1}/{n_episodes}: reward={total:.1f} flag={info.get('flag_get',False)}")

    env.close()
    print(
        f"\nResults over {n_episodes} episodes:\n"
        f"  mean={np.mean(rewards):.1f}  std={np.std(rewards):.1f}\n"
        f"  min={np.min(rewards):.1f}  max={np.max(rewards):.1f}\n"
        f"  flag-get rate={flags/n_episodes:.1%}"
    )


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def _run(cfg: "Config") -> None:  # noqa: F821
    from config import Config
    from environment import make_env
    from agent import Mario
    from metrics import MetricLogger
    from mcp_server import update_state

    _seed(cfg.train.seed)

    if cfg.multi_agent.enabled:
        from multi_agent import MultiAgentCoordinator
        sd = cfg.train.resolve_save_dir()
        MultiAgentCoordinator(cfg).run(cfg.train.episodes, sd)
        return

    sd = cfg.train.resolve_save_dir()
    sd.mkdir(parents=True, exist_ok=True)

    env = make_env(cfg.env, seed=cfg.train.seed)
    mario = Mario(
        state_dim=cfg.state_dim,
        action_dim=env.action_space.n,
        save_dir=sd,
        config=cfg.agent,
        checkpoint=cfg.train.checkpoint,
    )

    logger = MetricLogger(
        save_dir=sd,
        use_wandb=cfg.wandb.enabled,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        config={
            "env": cfg.env.__dict__,
            "agent": cfg.agent.__dict__,
        },
    )

    improver = None
    if cfg.autonomous.enabled:
        from autonomous import SelfImprovementLoop
        improver = SelfImprovementLoop(cfg.autonomous, cfg.env)

    update_state(
        running=True,
        config=cfg.agent.__dict__,
        checkpoint_dir=str(sd),
    )

    recent_rewards: list = []

    for episode in range(cfg.train.episodes):
        obs, _ = env.reset()
        flag_get = False
        total_reward = 0.0

        while True:
            action = mario.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            mario.cache(obs, next_obs, action, reward, done)
            q, loss = mario.learn()
            logger.log_step(reward, loss, q)
            total_reward += reward
            obs = next_obs
            if done or info.get("flag_get"):
                flag_get = bool(info.get("flag_get"))
                break

        logger.log_episode()
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        update_state(
            episode=episode,
            step=mario.curr_step,
            epsilon=mario.exploration_rate,
            mean_reward_100=float(np.mean(recent_rewards)) if recent_rewards else 0.0,
        )

        if improver:
            events = improver.on_episode_end(episode, total_reward, flag_get, mario)
            if "curriculum_advance" in events:
                world, stage = events["curriculum_advance"]
                cfg.env.world, cfg.env.stage = world, stage
                env.close()
                env = make_env(cfg.env, seed=cfg.train.seed + episode)
            update_state(
                curriculum_level=events.get("curriculum_level", (1, 1)),
                curriculum_success_rate=events.get("curriculum_success_rate", 0.0),
            )

        if episode % 20 == 0:
            logger.record(
                episode=episode,
                epsilon=mario.exploration_rate,
                step=mario.curr_step,
            )

    update_state(running=False)
    logger.close()
    env.close()


# Expose for import by tests / notebooks
import numpy as np  # noqa: E402 (needed in evaluate)


if __name__ == "__main__":
    app()
