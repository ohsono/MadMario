#!/usr/bin/env python3
"""MCP server — exposes MadMario training controls as Claude-callable tools.

Run standalone:  python mcp_server.py
Add to Claude Code .claude/settings.json:
  {
    "mcpServers": {
      "madmario": {
        "command": "python",
        "args": ["mcp_server.py"],
        "cwd": "/path/to/MadMario"
      }
    }
  }
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ---------------------------------------------------------------------------
# Shared training state — written by the training loop, read by MCP tools
# ---------------------------------------------------------------------------
_state: Dict[str, Any] = {
    "running": False,
    "episode": 0,
    "step": 0,
    "epsilon": 1.0,
    "mean_reward_100": 0.0,
    "checkpoint_dir": None,
    "curriculum_level": (1, 1),
    "curriculum_success_rate": 0.0,
    "config": {},
}
_state_lock = threading.Lock()

server = Server("madmario")


def update_state(**kwargs: Any) -> None:
    """Call from the training loop to push live metrics to MCP."""
    with _state_lock:
        _state.update(kwargs)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_training_status",
            description=(
                "Return live MadMario training metrics: episode count, global step, "
                "epsilon, 100-episode mean reward, curriculum level."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="list_checkpoints",
            description="List all saved .chkpt files under a directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Root directory to search (default: checkpoints/)",
                    }
                },
            },
        ),
        types.Tool(
            name="evaluate_checkpoint",
            description=(
                "Load a checkpoint and run N greedy evaluation episodes. "
                "Returns mean reward, std, and flag-get rate."
            ),
            inputSchema={
                "type": "object",
                "required": ["checkpoint_path"],
                "properties": {
                    "checkpoint_path": {"type": "string"},
                    "n_episodes": {
                        "type": "integer",
                        "default": 10,
                        "description": "Number of evaluation episodes",
                    },
                    "world": {"type": "integer", "default": 1},
                    "stage": {"type": "integer", "default": 1},
                },
            },
        ),
        types.Tool(
            name="get_curriculum_status",
            description="Return current curriculum level and per-level success rate.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_config",
            description="Return the active training hyperparameter configuration.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: Dict) -> list[types.TextContent]:
    if name == "get_training_status":
        with _state_lock:
            snapshot = {k: v for k, v in _state.items() if k != "config"}
        return [types.TextContent(type="text", text=json.dumps(snapshot, indent=2))]

    if name == "list_checkpoints":
        directory = Path(arguments.get("directory", "checkpoints"))
        if not directory.exists():
            return [types.TextContent(type="text", text="No checkpoints directory found.")]
        ckpts = sorted(directory.rglob("*.chkpt"))
        body = "\n".join(str(p) for p in ckpts) if ckpts else "No checkpoints found."
        return [types.TextContent(type="text", text=body)]

    if name == "evaluate_checkpoint":
        ckpt = Path(arguments["checkpoint_path"])
        n = int(arguments.get("n_episodes", 10))
        world = int(arguments.get("world", 1))
        stage = int(arguments.get("stage", 1))
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, _run_evaluation, ckpt, n, world, stage
        )
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "get_curriculum_status":
        with _state_lock:
            cur = {
                "level": _state["curriculum_level"],
                "success_rate": _state["curriculum_success_rate"],
            }
        return [types.TextContent(type="text", text=json.dumps(cur, indent=2))]

    if name == "get_config":
        with _state_lock:
            cfg = _state.get("config", {})
        return [types.TextContent(type="text", text=json.dumps(cfg, indent=2))]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


# ---------------------------------------------------------------------------
# Synchronous evaluation helper (runs in thread executor)
# ---------------------------------------------------------------------------

def _run_evaluation(
    ckpt_path: Path,
    n_episodes: int,
    world: int = 1,
    stage: int = 1,
) -> Dict[str, Any]:
    import numpy as np
    from madmario.config import Config, EnvConfig
    from madmario.environment import make_env
    from madmario.agent import Mario

    cfg = Config(env=EnvConfig(world=world, stage=stage))
    env = make_env(cfg.env)
    mario = Mario(
        state_dim=cfg.state_dim,
        action_dim=env.action_space.n,
        save_dir=ckpt_path.parent,
        config=cfg.agent,
        checkpoint=ckpt_path,
    )
    mario.exploration_rate = 0.0

    rewards, flags = [], 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        info: Dict = {}
        while not done:
            action = mario.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated or bool(info.get("flag_get"))
        rewards.append(total)
        if info.get("flag_get"):
            flags += 1
    env.close()

    return {
        "checkpoint": str(ckpt_path),
        "n_episodes": n_episodes,
        "world": world,
        "stage": stage,
        "mean_reward": round(float(np.mean(rewards)), 3),
        "std_reward": round(float(np.std(rewards)), 3),
        "min_reward": round(float(np.min(rewards)), 3),
        "max_reward": round(float(np.max(rewards)), 3),
        "flag_get_rate": flags / n_episodes,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Sync wrapper for the `madmario-mcp` console script."""
    asyncio.run(_main())


if __name__ == "__main__":
    main()
