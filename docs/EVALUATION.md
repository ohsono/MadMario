# MadMario — Multi-Agent v2 Evaluation

Comparison of the three training modes on `SuperMarioBros-1-1-v0`, CPU-only
(12 cores), seed 42, replay capacity 10 K, burnin 200. Single-agent and PBT
workers use fast ε-decay (0.9995/step) so learning is visible at this budget;
shared-mode actors use the fixed ε ladder by design.

W&B mirror: <https://wandb.ai/ohsono-private/madmario-eval> (run `o34fu8t2`).

## Setup

| Mode | Command (abridged) | Episodes | Processes |
|---|---|---|---|
| single | `train.py train --episodes 150 --eps-decay 0.9995` | 150 | 1 |
| shared | `train.py train --multi-agent --ma-mode shared --num-agents 4` | 148 (4×37) | 1 learner + 4 actors |
| pbt | `train.py train --multi-agent --ma-mode pbt --num-agents 4 --pbt-interval 10 --eps-decay 0.9995` | 148 (4×37) | 4 full agents |

## Results

| mode | episodes | mean reward | **final 20-ep mean** | wall (s) |
|---|---|---|---|---|
| single | 150 | 575.2 | 635.6 | 339 |
| **shared** | 148 | 465.0 | **903.9** | 730 |
| pbt | 148 | 584.3 | 605.0 | 563 |

(Random-policy baseline on this level is ≈230. No flag-gets at this budget —
finishing 1-1 typically needs ≳10K episodes.)

![Reward vs episodes](plots/reward_vs_episodes.png)

![Reward vs wall-clock](plots/reward_vs_wallclock.png)

## Interpretation

**Shared (Ape-X style) ends far ahead and is still climbing.** Its pooled
curve starts at the random baseline (~230) and lags for the first ~50
episodes — expected, since the curve averages *all* actors, including the
permanent ε=0.4 explorer, and the learner needs to fill the shared buffer
before its 15,005 gradient steps start paying off. From episode ~130 it
surges past both baselines to a 903.9 final mean, the strongest policy of the
three, with no sign of plateau at cutoff. The per-actor plot shows the
mechanism: the near-greedy actors (ε=0.016, ε=0.003) track the learner's
improving policy, while the explorer keeps feeding diverse transitions.

![Per-actor rewards, shared mode](plots/per_agent_shared.png)

**Single agent learns fastest initially, then oscillates.** With fast ε-decay
it exploits early (peak ~780 by episode 10) but churns between 350–650
afterwards — the classic instability of one annealed exploration schedule on
one data stream.

**PBT demonstrates its mechanism at this budget rather than a score win.**
Three exploit events fired, each correctly targeting the bottom-quartile
agent and copying a top-quartile member's weights + perturbed
hyperparameters (γ explored to 0.997 and 0.72; lr to 2.0e-4). The
hyperparameter trajectories below are the real deliverable — PBT's payoff
grows with budget, and 37 episodes/agent only allows 3 selection rounds.

![Per-agent rewards, PBT](plots/per_agent_pbt.png)

![PBT hyperparameter trajectories](plots/pbt_hparams.png)

## Caveats

- Single seed, ~150 episodes/mode, CPU-only: these measure **early learning
  dynamics**, not final performance (Henderson et al. 2018 — see
  [THEORY.md](THEORY.md) §6). Treat rankings as indicative.
- Wall-clock is not matched across modes (339–730 s); the shared learner did
  ~3× the gradient steps of the single agent in ~2× the time on the same data
  budget — its advantage compounds on the wall-clock axis at larger scale.
- Shared-mode "mean reward" (465.0) is structurally depressed by the
  high-ε actors; the final 20-ep mean (pooled, time-ordered) and the
  near-greedy actors' curves are the fair policy-quality measures.

## Reproduce

```bash
python train.py train --episodes 150 --seed 42 --burnin 200 \
    --memory-size 10000 --eps-decay 0.9995 --save-dir runs/single
python train.py train --multi-agent --ma-mode shared --num-agents 4 \
    --episodes 150 --seed 42 --burnin 200 --memory-size 10000 --save-dir runs/shared
python train.py train --multi-agent --ma-mode pbt --num-agents 4 --pbt-interval 10 \
    --episodes 150 --seed 42 --burnin 200 --memory-size 10000 \
    --eps-decay 0.9995 --save-dir runs/pbt
python plot_compare.py runs/single runs/shared runs/pbt \
    --out docs/plots --wandb-project madmario-eval
```
