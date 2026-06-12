#!/usr/bin/env python3
"""Comparison plots for MadMario runs.

Reads the `history.json` each training mode writes into its save dir and
produces PNG learning-curve comparisons (and optionally mirrors them to W&B).

Usage:
    python plot_compare.py runs/single runs/shared runs/pbt \
        --out docs/plots --wandb-project madmario-eval
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer(add_completion=False)

LABELS = {
    "single": "Single agent (Double DQN)",
    "shared": "Multi-agent shared (Ape-X style)",
    "pbt": "Multi-agent PBT",
}
COLORS = {"single": "tab:blue", "shared": "tab:red", "pbt": "tab:green"}


def _load(run_dir: Path) -> Dict:
    data = json.loads((run_dir / "history.json").read_text())
    data["episodes"].sort(key=lambda e: e.get("wall", 0.0))
    return data


def _rolling(x: List[float], w: int = 20) -> np.ndarray:
    if not x:
        return np.array([])
    out = np.empty(len(x))
    for i in range(len(x)):
        out[i] = float(np.mean(x[max(0, i - w + 1) : i + 1]))
    return out


def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)


@app.command()
def main(
    run_dirs: List[Path] = typer.Argument(..., help="Dirs containing history.json"),
    out: Path = typer.Option(Path("docs/plots"), help="Output dir for PNGs"),
    window: int = typer.Option(20, help="Rolling-mean window"),
    wandb_project: Optional[str] = typer.Option(None, help="Mirror plots to W&B"),
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    runs = {(_load(d))["mode"]: _load(d) for d in run_dirs}
    pngs: List[Path] = []

    # 1. Reward vs total episodes (sample efficiency)
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode, data in runs.items():
        rewards = [e["reward"] for e in data["episodes"]]
        ax.plot(
            _rolling(rewards, window),
            label=LABELS.get(mode, mode),
            color=COLORS.get(mode),
        )
    _style(ax, "Total episodes (all agents pooled, time-ordered)",
           f"Reward ({window}-ep rolling mean)",
           "Sample efficiency — reward vs episodes")
    p = out / "reward_vs_episodes.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); pngs.append(p)

    # 2. Reward vs wall-clock (systems efficiency)
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode, data in runs.items():
        eps = data["episodes"]
        rewards = _rolling([e["reward"] for e in eps], window)
        wall = [e.get("wall", i) for i, e in enumerate(eps)]
        ax.plot(wall, rewards, label=LABELS.get(mode, mode), color=COLORS.get(mode))
    _style(ax, "Wall-clock time (s)", f"Reward ({window}-ep rolling mean)",
           "Systems efficiency — reward vs wall-clock")
    p = out / "reward_vs_wallclock.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); pngs.append(p)

    # 3. Per-agent curves for each multi-agent run
    for mode, data in runs.items():
        agents = sorted({e["agent_id"] for e in data["episodes"]})
        if len(agents) < 2:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for aid in agents:
            rs = [e["reward"] for e in data["episodes"] if e["agent_id"] == aid]
            extra = ""
            if mode == "shared":
                eps_vals = [e.get("epsilon") for e in data["episodes"]
                            if e["agent_id"] == aid and e.get("epsilon") is not None]
                if eps_vals:
                    extra = f" (ε={eps_vals[0]:.3g})"
            ax.plot(_rolling(rs, max(5, window // 2)), label=f"agent {aid}{extra}")
        _style(ax, "Episode (per agent)", "Reward (rolling mean)",
               f"Per-agent rewards — {LABELS.get(mode, mode)}")
        p = out / f"per_agent_{mode}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); pngs.append(p)

    # 4. PBT hyperparameter trajectories
    if "pbt" in runs:
        eps = runs["pbt"]["episodes"]
        agents = sorted({e["agent_id"] for e in eps})
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for aid in agents:
            mine = [e for e in eps if e["agent_id"] == aid and "hparams" in e]
            axes[0].plot([e["hparams"]["lr"] for e in mine], label=f"agent {aid}")
            axes[1].plot([e["hparams"]["gamma"] for e in mine], label=f"agent {aid}")
        axes[0].set_yscale("log")
        _style(axes[0], "Episode", "learning rate", "PBT lr trajectories")
        _style(axes[1], "Episode", "gamma", "PBT gamma trajectories")
        p = out / "pbt_hparams.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); pngs.append(p)

    # 5. Summary table
    print(f"\n{'mode':<10}{'episodes':>10}{'mean reward':>14}"
          f"{'final 20-ep mean':>18}{'flags':>8}{'wall (s)':>10}")
    for mode, data in runs.items():
        eps = data["episodes"]
        rewards = [e["reward"] for e in eps]
        flags = sum(bool(e.get("flag_get")) for e in eps)
        wall = max((e.get("wall", 0.0) for e in eps), default=0.0)
        print(f"{mode:<10}{len(eps):>10}{np.mean(rewards):>14.1f}"
              f"{np.mean(rewards[-20:]):>18.1f}{flags:>8}{wall:>10.0f}")

    print(f"\nPlots written to {out}/:")
    for p in pngs:
        print(f"  {p.name}")

    if wandb_project:
        import wandb
        run = wandb.init(project=wandb_project, name="comparison", job_type="eval")
        for mode, data in runs.items():
            tbl = wandb.Table(columns=["episode", "reward", "wall", "agent_id"])
            for i, e in enumerate(data["episodes"]):
                tbl.add_data(i, e["reward"], e.get("wall", 0.0), e["agent_id"])
            run.log({f"{mode}/history": tbl})
        run.log({p.stem: wandb.Image(str(p)) for p in pngs})
        run.finish()
        print(f"Mirrored to W&B project '{wandb_project}'.")


if __name__ == "__main__":
    app()
