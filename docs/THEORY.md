# MadMario — Underlying Theory & Related Work

The math and literature behind each component of the codebase, from the MDP
formulation to the distributed/multi-agent methods used in v2.

---

## 1. Problem formulation: Mario as an MDP

Super Mario Bros is modeled as a Markov Decision Process (S, A, P, R, γ):

- **State** s ∈ ℝ^(4×84×84): four stacked, grayscale, downsampled frames.
  Stacking restores (approximate) Markovness — a single frame hides velocity;
  four frames let a feedforward net infer it (Mnih et al. 2015).
- **Actions** A: a reduced discrete set (`[["right"], ["right","A"]]` by
  default) via `JoypadSpace` — restricting the action space is itself a
  domain-knowledge prior that drastically shrinks exploration.
- **Reward** R: the `gym-super-mario-bros` shaped reward
  r = Δx (progress) + Δclock (time penalty) + death penalty, clipped to
  [−15, 15]. Shaped rewards make the 1-1 task tractable for DQN but are a
  known source of reward-hacking risk on later levels.
- **Frame skip** (4): the agent repeats each action 4 frames
  (`SkipFrame`), summing rewards. This is the standard Atari action-repeat
  trick — ~4× cheaper stepping with negligible control loss.
- **Discount** γ = 0.9.

The objective is the optimal action-value function

  Q*(s, a) = E[ Σₜ γᵗ rₜ | s₀ = s, a₀ = a, π* ].

## 2. Q-learning and DQN

Tabular Q-learning (Watkins & Dayan 1992) iterates the Bellman optimality
backup; with function approximation the update becomes regression toward the
TD target. **DQN** (Mnih et al. 2015, [Nature 518](https://www.nature.com/articles/nature14236))
made this stable for CNNs with two mechanisms, both present in `agent.py`:

1. **Experience replay** (Lin 1992): transitions (s, a, r, s′, done) stored in
   a buffer and sampled i.i.d., breaking the temporal correlation that
   destabilizes SGD, and reusing each transition multiple times
   (`replay.py`, capacity 100 K, `learn_every=3`).
2. **Target network**: a frozen copy Q_θ⁻ provides TD targets, updated only
   every `sync_every=10 000` steps — without it the regression target moves
   with every gradient step (the "deadly triad": function approximation +
   bootstrapping + off-policy, Sutton & Barto 2018, ch. 11).

Loss (Huber/SmoothL1, as in the original DQN for gradient clipping behavior):

  L(θ) = E[ ℓ( Q_θ(s,a) − y ) ],  y = r + γ (1−done) max_{a′} Q_θ⁻(s′, a′).

## 3. Double DQN — what MadMario actually runs

The max operator in the DQN target both *selects* and *evaluates* with the
same noisy estimator, biasing targets upward (overestimation; Thrun &
Schwartz 1993). **Double DQN** (van Hasselt, Guez & Silver 2015,
[arXiv:1509.06461](https://arxiv.org/abs/1509.06461)) decouples them:

  y^DDQN = r + γ (1−done) · Q_θ⁻(s′, **argmax_{a′} Q_θ(s′, a′)**)

— the *online* net picks the action, the *target* net scores it
(`agent.py:td_target`). `MarioNet` (`neural.py`) holds both copies of the same
CNN (Conv 32-8×8-s4 → Conv 64-4×4-s2 → Conv 64-3×3-s1 → FC 512 → |A|, the
classic Nature-DQN trunk) with the target branch's parameters frozen
(`requires_grad=False`) and synced by `load_state_dict`.

**ε-greedy exploration**: behavior policy is uniform-random with probability ε,
greedy otherwise; ε decays 1.0 → 0.1 multiplicatively per step. Because
Q-learning is **off-policy**, learning from ε-greedy (or any other) behavior
data toward the greedy target policy is sound — this is the property the
v2 multi-agent design exploits (§5).

## 4. Curriculum learning and self-improvement (phase 2)

- **Curriculum learning** (Bengio et al. 2009,
  [ICML](https://dl.acm.org/doi/10.1145/1553374.1553380)): ordering tasks
  easy→hard accelerates and regularizes learning. In RL specifically, see the
  survey by Narvekar et al. 2020 ([arXiv:2003.04960](https://arxiv.org/abs/2003.04960)).
  `CurriculumManager` implements a *fixed-order, mastery-gated* curriculum
  over the 32 (world, stage) pairs: advance when the flag-get rate over a
  100-episode window reaches 70 %. This is the simplest member of the
  curriculum family — task sequencing is hand-specified; only the *pacing* is
  learned from performance.
- **Plateau-driven exploration boost**: `PlateauDetector` lowers ε_min when the
  best episode reward stagnates for 200 episodes — a small instance of
  *adaptive exploration scheduling*; conceptually related to ε-scheduling
  studies and to count/novelty-based exploration as the principled extreme
  (Bellemare et al. 2016, [arXiv:1606.01868](https://arxiv.org/abs/1606.01868)).

## 5. Distributed and multi-agent deep RL — the v2 lineage

The v2 redesign follows a well-trodden lineage of *parallelizing a single
learning problem*:

| System | Year | Key idea | What MadMario v2 borrows |
|---|---|---|---|
| **Gorila** (Nair et al., [arXiv:1507.04296](https://arxiv.org/abs/1507.04296)) | 2015 | Distributed DQN: parallel actors *and* parallel learners with a parameter server | the actor/learner role separation |
| **A3C** (Mnih et al., [arXiv:1602.01783](https://arxiv.org/abs/1602.01783)) | 2016 | Asynchronous actor-critic workers, gradient sharing, *no replay* | evidence that parallel actors substitute for replay diversity (we keep replay since DQN is off-policy) |
| **Ape-X** (Horgan et al., [arXiv:1803.00933](https://arxiv.org/abs/1803.00933)) | 2018 | Many cheap actors → **one shared prioritized replay** → one GPU learner; **per-actor fixed ε ladder** εᵢ = ε·α^(i/(N−1)) | the entire shared-mode architecture and the ε ladder (ε=0.4, α=1/128); we omit prioritization (future work) |
| **IMPALA** (Espeholt et al., [arXiv:1802.01561](https://arxiv.org/abs/1802.01561)) | 2018 | Actor–learner at datacenter scale; V-trace off-policy correction | scale ceiling of the pattern; V-trace unnecessary for value-based DQN |
| **R2D2** (Kapturowski et al., [ICLR 2019](https://openreview.net/forum?id=r1lyTjAqYX)) | 2019 | Recurrent replay in the Ape-X frame | listed as future work for long-horizon levels |

**Why sharing experience is sound here:** Q-learning's update does not require
that the data come from the learner's own policy — only adequate state-action
coverage. Transitions generated by N actors with different ε (or slightly
stale weights) are simply a richer behavior distribution. This is precisely
why Ape-X outperformed single-actor DQN at *equal* learner throughput: better
data, not more gradients.

**Population-Based Training** (Jaderberg et al. 2017,
[arXiv:1711.09846](https://arxiv.org/abs/1711.09846)) is the second pillar of
v2: treat the population as an online hyperparameter optimizer. Each member
trains normally; periodically, underperformers **exploit** (copy weights +
hyperparameters from a top performer) and **explore** (perturb the copied
hyperparameters ×0.8/×1.2). Unlike grid/random search, PBT searches
*schedules* (the effective hyperparameter is a trajectory, not a point) at no
extra compute beyond the population itself. v2's PBT mode implements the
canonical truncation-selection variant: bottom quartile exploits a random
top-quartile member.

**Contrast with "true" multi-agent RL (MARL).** MadMario v2 is *distributed
single-agent RL*: agents share a task but do not interact in the environment.
Genuine MARL — non-stationarity from co-adapting agents, credit assignment,
emergent communication — is a different problem class (see Lowe et al. 2017,
MADDPG, [arXiv:1706.02275](https://arxiv.org/abs/1706.02275); OpenAI Five;
AlphaStar league training, Vinyals et al. 2019). Mario is single-player, so
the population/parallelism framing is the correct one; the league-style
*population* ideas (AlphaStar) are nonetheless the conceptual ancestor of the
specialist→generalist distillation sketched in COMPARISON.md §5.3.

## 6. Evaluation methodology

- **Metrics:** per-episode reward, 20/100-episode rolling mean, episode
  length, TD loss, mean Q, ε, flag-get rate. Rolling means are reported
  because single-episode reward in Mario has high variance (death is bimodal).
- **Axes:** curves are plotted against *episodes* (sample efficiency) and
  *wall-clock* (systems efficiency) — distributed methods can win the second
  while tying the first, and reporting only one axis is a classic deep-RL
  evaluation pitfall (Henderson et al. 2018,
  [arXiv:1709.06560](https://arxiv.org/abs/1709.06560)).
- **Caveats:** CPU-only short runs measure *early learning dynamics*, not
  final performance; seeds are fixed and reported; deep-RL results at this
  budget are indicative, not conclusive ("Deep RL Doesn't Work Yet",
  Irpan 2018).

## 7. Reference list

1. Mnih et al., *Human-level control through deep reinforcement learning*, Nature 2015.
2. van Hasselt, Guez, Silver, *Deep RL with Double Q-learning*, AAAI 2016 (arXiv:1509.06461).
3. Lin, *Self-improving reactive agents based on RL, planning and teaching*, MLJ 1992.
4. Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018.
5. Bengio et al., *Curriculum Learning*, ICML 2009.
6. Narvekar et al., *Curriculum Learning for RL Domains: A Framework and Survey*, JMLR 2020 (arXiv:2003.04960).
7. Nair et al., *Massively Parallel Methods for Deep RL (Gorila)*, 2015 (arXiv:1507.04296).
8. Mnih et al., *Asynchronous Methods for Deep RL (A3C)*, ICML 2016 (arXiv:1602.01783).
9. Horgan et al., *Distributed Prioritized Experience Replay (Ape-X)*, ICLR 2018 (arXiv:1803.00933).
10. Espeholt et al., *IMPALA: Scalable Distributed Deep-RL*, ICML 2018 (arXiv:1802.01561).
11. Kapturowski et al., *Recurrent Experience Replay in Distributed RL (R2D2)*, ICLR 2019.
12. Jaderberg et al., *Population Based Training of Neural Networks*, 2017 (arXiv:1711.09846).
13. Lowe et al., *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)*, NeurIPS 2017 (arXiv:1706.02275).
14. Vinyals et al., *Grandmaster level in StarCraft II using multi-agent RL (AlphaStar)*, Nature 2019.
15. Henderson et al., *Deep RL That Matters*, AAAI 2018 (arXiv:1709.06560).
16. Schaul et al., *Prioritized Experience Replay*, ICLR 2016 (arXiv:1511.05952).
