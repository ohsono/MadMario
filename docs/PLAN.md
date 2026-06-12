# MadMario — Project Plan & Revision History

**Objective (current):** evolve MadMario from a single-agent Double DQN tutorial
into a **multi-agent reinforcement-learning system** — shared-experience
actor–learner training (Ape-X style) with population-based hyperparameter
search (PBT) — measured against the single-agent baseline with reproducible
plots and W&B dashboards.

---

## 1. Where the project started

The original codebase was a near-verbatim copy of the
[PyTorch Mario RL tutorial](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)
(2020): one Jupyter-style script training a Double DQN on `SuperMarioBros-1-1-v0`
with `gym==0.17`, no tests, no CLI, no logging beyond matplotlib files, and
eight latent bugs (see [COMPARISON.md](COMPARISON.md) §2) — the worst of which
stored CUDA tensors inside the replay buffer (~22 GB VRAM for a full buffer).

## 2. Revision history

| Phase | Date | Scope | Outcome |
|-------|------|-------|---------|
| **0 — Audit & bug fixes** | 2026-06 | Run existing code, identify failures | All 8 bugs fixed in `agent.py` / `replay.py` / `neural.py` / `metrics.py`; each fix locked by a regression test |
| **0.5 — Gymnasium migration** | 2026-06 | `gym 0.17` → `gymnasium ≥0.29` | New 5-tuple step API, `(obs, info)` reset, built-in `GrayscaleObservation` / `ResizeObservation` / `FrameStackObservation`; custom wrappers reduced to `SkipFrame` only |
| **1 — Observability** | 2026-06 | wandb integration | `MetricLogger` streams per-step loss/Q and per-episode reward/length/ε to W&B; local matplotlib plots retained as fallback |
| **2 — Autonomous RL** | 2026-06 | Self-improvement loop | `CurriculumManager` (32-level world/stage progression, advance at ≥70 % flag rate over 100 episodes), `PlateauDetector` (ε-min annealing on stagnation) |
| **3 — MCP server** | 2026-06 | Claude Code integration | `mcp_server.py` with 5 read tools: training status, checkpoints, evaluation, curriculum, config |
| **4 — Harness engineering** | 2026-06 | Tests, CLI, config | 35 pytest tests; Typer CLI (`train` / `evaluate`); dataclass `Config` replacing magic numbers; `pyproject.toml` |
| **5 — Multi-agent v1** | 2026-06 | Population training | `MultiAgentCoordinator`: N independent workers, best-weight broadcast every 50 episodes |
| **6 — Multi-agent v2** | 2026-06 (this revision) | Actor–learner + PBT | Shared replay (Ape-X style), per-actor ε ladder, PBT exploit/explore, evaluation harness with comparison plots |

## 3. Why v1 multi-agent had to be redesigned

Phase 5's population design had three structural flaws (detailed analysis in
[COMPARISON.md](COMPARISON.md) §4):

1. **No experience sharing** — N agents each kept private replay buffers and
   learners, so N processes mostly collected redundant data and split the
   compute budget N ways.
2. **Greedy weight broadcast collapsed diversity** — after the first broadcast
   the population converged to clones of one early-lucky agent; the `best_mean`
   high-water mark never decayed, so leadership could never change hands fairly.
3. **No division of labor** — every agent had identical hyperparameters and an
   identical task; the population searched nothing.

## 4. Multi-agent v2 plan (this revision)

### 4.1 Architecture

```
                       ┌──────────────────────────────┐
                       │      Learner (main proc)     │
                       │  shared ReplayBuffer (100 K) │
                       │  Double DQN gradient updates │
                       └───────┬──────────▲───────────┘
                  weights every│          │ transitions
                  N learn steps│          │ (mp.Queue)
              ┌────────────────┼──────────┼────────────────┐
              ▼                ▼          │                 ▼
        ┌──────────┐    ┌──────────┐      │          ┌──────────┐
        │ Actor 0  │    │ Actor 1  │     ...         │ Actor N-1│
        │ ε = 0.40 │    │ ε = 0.22 │                 │ ε = 0.012│
        │ own env  │    │ own env  │                 │ own env  │
        └──────────┘    └──────────┘                 └──────────┘
```

- **Shared mode (`--ma-mode shared`, default):** actors only step environments
  and select ε-greedy actions; all transitions flow to one central learner that
  owns the single replay buffer and does every gradient update. Actor *i* uses a
  fixed exploration rate εᵢ = ε·α^(i/(N−1)) (Ape-X ladder, ε = 0.4, α = 1/128),
  giving persistent, structured exploration diversity instead of accidental
  seed diversity.
- **PBT mode (`--ma-mode pbt`):** N full agents as in v1, but sync events are
  *exploit + explore*: agents in the bottom quartile (by **current** rolling
  mean, re-ranked each event) copy a top-quartile member's weights **and**
  hyperparameters, then perturb `lr` and `gamma` by ×0.8/×1.2. The population
  becomes an online hyperparameter search rather than a clone factory.

### 4.2 Milestones

| # | Milestone | Acceptance criterion |
|---|-----------|----------------------|
| M1 | Bug-free single-agent loop | `train.py train --episodes 30 --burnin 0` completes; checkpoints + plots produced |
| M2 | `multi_agent.py` v2 shared mode | N actors feed one learner; weights round-trip verified; no queue deadlock at shutdown |
| M3 | `multi_agent.py` v2 PBT mode | Exploit/explore events logged; per-agent hyperparameters diverge over time |
| M4 | Tests | New tests for ε-ladder, PBT perturbation, transition routing; full suite green |
| M5 | Evaluation | Single-agent vs shared-mode learning curves on World 1-1; PNG plots in `docs/plots/`; W&B runs when credentials available |
| M6 | Documentation | This file, COMPARISON.md, THEORY.md committed |

### 4.3 Out of scope (future work)

- Prioritized experience replay (the natural next step after Ape-X plumbing).
- Recurrent value heads (R2D2) for partial observability beyond frame stacking.
- Specialist-per-world population with policy distillation into a generalist
  (sketched in COMPARISON.md §5.3).
- LLM-as-meta-agent: write-capable MCP tools (`set_hyperparameter`,
  `advance_curriculum`) turning the MCP server from a dashboard into a control
  plane.

## 5. Constraints

- **Hardware:** CPU-only (12 cores) on the development machine — evaluation
  runs use small episode budgets; the architecture is GPU-ready (learner moves
  batches to `cuda` automatically when available).
- **Repository:** SHA-1 GitLab repo `ohsono-group/madmario`; commits carry no
  AI co-author tags per project policy. The `hs205stats404/Final/MadMario`
  copy is synced by rsync (SHA-256/SHA-1 submodule incompatibility).
