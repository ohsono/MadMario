# MadMario — In-Depth Comparison: Original → Modernized → Multi-Agent v2

This document compares the three generations of the codebase:

- **Gen 1 — Original** (PyTorch tutorial port, `gym 0.17`, single script)
- **Gen 2 — Modernized** (phases 0–5: bug fixes, gymnasium, wandb, curriculum,
  MCP, harness, population multi-agent v1)
- **Gen 3 — Multi-agent v2** (this revision: shared-experience actor–learner
  + population-based training)

---

## 1. Architecture at a glance

| Dimension | Gen 1 (original) | Gen 2 (modernized) | Gen 3 (multi-agent v2) |
|---|---|---|---|
| Entry point | `main.py` monolith | Typer CLI (`train` / `evaluate`) | same CLI, `--multi-agent --ma-mode shared\|pbt` |
| Env API | `gym 0.17` (4-tuple step) | `gymnasium ≥0.29` (5-tuple step) | same |
| Config | magic numbers inline | dataclass `Config` tree | + `MultiAgentConfig.mode`, ε-ladder, PBT knobs |
| Replay buffer | CUDA tensors in deque | CPU `float32` numpy, per-batch transfer | **one shared buffer in the learner**, fed by N actors |
| Learners | 1 | 1 (or N independent in v1) | **exactly 1** (shared) or N with PBT search |
| Exploration | single decaying ε | single decaying ε per agent | **fixed per-actor ε ladder** (shared) / per-agent decaying ε (PBT) |
| Observability | matplotlib files | + wandb streaming | + per-actor/per-agent metrics, comparison plots |
| Tests | none | 35 pytest tests | + multi-agent unit tests |
| Self-improvement | none | curriculum + plateau detection | composes with both modes |
| External control | none | MCP server (5 read tools) | same (write tools = future work) |

---

## 2. Gen 1 → Gen 2: the eight bugs, and why each mattered

| # | Bug (Gen 1) | Consequence | Fix (Gen 2) |
|---|---|---|---|
| 1 | Replay deque stored **CUDA tensors** | ~22 GB VRAM at 100 K capacity → OOM on any consumer GPU; silently forced tiny buffers | Buffer stores `float32` numpy on CPU; only the 32-sample batch moves to device (`replay.py`) |
| 2 | `curr_step` absent from checkpoints | Resumed runs repeated the 100 K-step burnin and re-decayed ε from the wrong point | `curr_step` saved/restored (`agent.py:save/load`) |
| 3 | Optimizer state never saved | Adam's first/second-moment estimates reset on resume → loss spike, effective LR jump | `optimizer.state_dict()` in every checkpoint |
| 4 | `sync_every`/`save_every` checked **before** burnin guard | Target net synced to random online weights during burnin; useless checkpoints written | Burnin guard first in `learn()` |
| 5 | Rewards stored as `DoubleTensor` | Silent float64 upcast through the TD target → slower CPU math, dtype mismatch risk | `float32` end-to-end |
| 6 | TD indexing used hardcoded `batch_size` | Crash (or silent mis-indexing) whenever the sampled batch ≠ 32, e.g. final partial batches | `torch.arange(state.shape[0])` |
| 7 | `MarioNet.forward` returned `None` on a typo'd model name | `NoneType` error surfaced far from the cause | Explicit `ValueError` |
| 8 | `if loss:` dropped `loss == 0.0` | Metrics under-counted learning steps exactly when the model fit a batch perfectly | `if loss is not None:` |

A ninth issue was found during v2 work: `learn()` had **no minimum-buffer
guard**, so `--burnin 0` crashed with `ValueError: Sample larger than
population`. Fixed with `len(self.memory) < batch_size → skip`.

**API migration.** `gym_super_mario_bros 9.x` internally uses
`gymnasium.spaces.Box`, so Gen 1's `gym.wrappers` failed `isinstance` checks
outright — the original code no longer ran at all on a current install.
Gen 2 rebuilt the pipeline on gymnasium built-ins
(`GrayscaleObservation → ResizeObservation → TransformObservation →
FrameStackObservation`), keeping only `SkipFrame` as custom code (~75 % less
wrapper code to maintain).

---

## 3. Gen 2 baseline: what single-agent Double DQN gives us

The modernized single agent is a faithful Double DQN (van Hasselt et al. 2015,
see [THEORY.md](THEORY.md) §3):

- online net selects the argmax action, frozen target net evaluates it —
  decoupling selection from evaluation to fight Q-value overestimation;
- 84×84×4 stacked grayscale frames, 4-frame action repeat;
- ε-greedy from 1.0 → 0.1 with decay 0.99999975/step (~9.2 M steps to floor).

Its limits on this problem:

- **Sample inefficiency.** One environment stream means wall-clock time is
  dominated by env stepping; on CPU the single process can't saturate even one
  core's worth of learner compute.
- **Exploration fragility.** A single annealed ε schedule explores a lot early
  and almost nothing late; if the policy lands in a local optimum (e.g. always
  dying at the same pipe), nothing reinjects diversity except the plateau
  detector's ε-min nudge.
- **Hyperparameter sensitivity.** `lr`, `gamma`, decay rate are fixed guesses;
  DQN on a new task typically needs a sweep, which a single run cannot do.

---

## 4. Gen 2 multi-agent v1 vs Gen 3 v2 — the core of this revision

### 4.1 What v1 actually did

```
N × (env + ε-greedy + private buffer + private learner)
          │  episode results + occasional weights
          ▼
   coordinator: if 20-ep mean > all-time best → broadcast weights to all
```

### 4.2 Why that design underdelivers

| Flaw | Mechanism | Effect |
|---|---|---|
| **Redundant data** | N private buffers on the same level, near-identical policies after first broadcast | ~N× env compute for ≈1× unique experience |
| **Diversity collapse** | best-weight broadcast overwrites every other policy | population degenerates to clones; exploration benefit lost after ~1 sync interval |
| **Stale leadership** | `best_mean` was an all-time high-water mark, never re-evaluated | an early lucky agent broadcast forever; later genuinely-better agents could not win |
| **Wasted bandwidth** | every worker serialized its full `state_dict` through the queue every sync interval, win or lose | (N−1)/N of weight payloads discarded |
| **Split compute** | N learners each doing gradient updates on 1/N of the data | no learner ever sees the pooled experience |

### 4.3 The v2 redesign

**Shared mode (default) — Ape-X-style actor–learner:**

| Property | v1 | v2 shared |
|---|---|---|
| Replay buffers | N private | **1 shared**, owned by the learner |
| Gradient updates | N learners × small data | 1 learner × pooled data (N× throughput into one buffer) |
| Exploration | identical decaying ε ×N | **ε ladder**: εᵢ = 0.4 · (1/128)^(i/(N−1)) — actor 0 explores hard forever, actor N−1 exploits |
| Weight flow | best→all, occasionally | learner→all actors, every `weight_sync_steps` learn steps |
| Transition flow | none between agents | actors → learner via `mp.Queue` |
| Off-policy correctness | n/a | DQN is off-policy: learning from other actors' behavior policies is *by design* sound |

Concretely: the actor process shrinks to *reset → act(ε fixed) → step → put
transition on queue → maybe load fresh weights*. All learning lives in the main
process. This converts the multi-process budget from "N weak learners" into
"N data collectors for one strong learner" — the same architectural move that
took DQN from Gorila to Ape-X (THEORY.md §5).

**PBT mode — population as hyperparameter search:**

| Property | v1 broadcast | v2 PBT |
|---|---|---|
| Sync trigger | new all-time best | fixed cadence, **re-ranked every event** on current rolling mean |
| Who changes | everyone except best | only **bottom quartile** |
| What is copied | weights only | weights **+ hyperparameters** of a random top-quartile member |
| After copy | nothing | **perturb**: `lr` and `gamma` ×0.8 or ×1.2 (clamped) |
| Population end-state | clones | a *distribution* over hyperparameters biased toward what works |

PBT keeps the top 75 % of the population untouched at every event, so
diversity survives; and because hyperparameters travel with weights, the
population performs an online schedule search (e.g. it can discover LR decay
emergently) that no single run can.

### 4.4 Expected empirical differences

On CPU with small budgets (what the evaluation in `docs/plots/` measures):

- **v2 shared vs single agent, equal wall-clock:** the shared learner sees ~N×
  transitions per second, so the learning curve in *wall-clock time* should
  dominate the single agent's; per *gradient step* the curves should be
  similar (same algorithm, better-filled buffer earlier).
- **v2 shared vs v1, equal episode budget:** v1 splits learning N ways and
  destroys exploration diversity, so v2 should reach a given mean reward in
  fewer total episodes.
- **PBT:** with tiny budgets PBT mostly demonstrates mechanism (hyperparameter
  divergence + exploit events) rather than final-score gains — PBT's payoff
  scales with budget. The evaluation reports both score curves and the
  hyperparameter trajectories.

---

## 5. What v2 deliberately does not do (and what would come next)

1. **Prioritized replay.** Ape-X pairs the actor–learner split with prioritized
   sampling (TD-error-proportional). The shared buffer is the prerequisite; the
   priority heap is the natural next PR.
2. **Recurrent agents (R2D2).** Frame stacking approximates a 4-step memory;
   an LSTM head would handle longer dependencies (e.g. maze levels in world 7).
3. **Specialist→generalist distillation.** Assign each population member a
   different world via `CurriculumManager`, then distill Q-values into one
   generalist network — the most promising direction for cross-level
   generalization.
4. **LLM meta-agent.** The MCP server currently only reads. Write tools
   (`set_hyperparameter`, `advance_curriculum`, `pause_agent`) would let
   Claude supervise training — detecting divergence and intervening — turning
   "autonomous RL" into a two-level agentic system: RL agents in the loop,
   an LLM agent above the loop.
