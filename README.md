# MadMario

Double DQN agent that learns to play Super Mario Bros — modernised from the
[PyTorch tutorial](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)
with bug fixes, wandb observability, autonomous curriculum learning, MCP server
integration, multi-agent training (shared-experience actor-learner + PBT), and
a full pytest harness.

📚 **Docs:** [project plan & revision history](docs/PLAN.md) ·
[in-depth generation comparison](docs/COMPARISON.md) ·
[theory & related work](docs/THEORY.md) ·
[evaluation results & plots](docs/EVALUATION.md)

---

## Quick start

```bash
pip install -r requirements.txt        # or: pip install -e ".[dev]"
python train.py train                  # standard 40 k-episode run
python train.py train --episodes 500   # short smoke run
python train.py evaluate checkpoints/.../mario_net_1.chkpt
```

---

## Installation

```bash
# Python 3.13+ REQUIRED — gym-super-mario-bros>=8.0 / nes-py>=9.0 (the
# gymnasium-API releases) only support Python 3.13+. On older Pythons pip
# will fail with "No matching distribution found for gym-super-mario-bros>=9.0".
pip install -r requirements.txt
```

Key dependencies: `torch`, `gymnasium`, `gym-super-mario-bros`, `wandb`,
`typer`, `mcp`, `opencv-python-headless`.

> **Note:** the old `environment.yml` (Python 3.8 / gym 0.17) is kept for
> reference only. The active codebase targets Python 3.13+ / gymnasium 0.29+.

---

## Training modes

### Standard
```bash
python train.py train
python train.py train --episodes 1000 --seed 7 --world 1 --stage 2
python train.py train --checkpoint checkpoints/2024-01-01T12-00-00/mario_net_1.chkpt
```

### With W&B observability
```bash
python train.py train --use-wandb --wandb-project my-mario-run
```
Streams reward, loss, Q-value, and epsilon live to your W&B dashboard.

### Autonomous curriculum
```bash
python train.py train --autonomous
```
Mario auto-advances through World 1-1 → … → 8-4 when flag-get rate ≥ 70 %
over the last 100 episodes. Plateau detection lowers ε-min if reward stagnates.

### Multi-agent v2 (see `docs/` for design & theory)
```bash
# Shared-experience actor-learner (Ape-X style) — default mode
python train.py train --multi-agent --ma-mode shared --num-agents 4 --episodes 5000

# Population Based Training (PBT) — online hyperparameter search
python train.py train --multi-agent --ma-mode pbt --num-agents 4 --pbt-interval 25
```
**shared:** N lightweight actors with a fixed per-actor ε ladder
(εᵢ = 0.4·(1/128)^(i/(N−1))) stream transitions to ONE learner that owns the
single replay buffer and does every gradient update, broadcasting fresh
weights back. **pbt:** N full agents; bottom-quartile members periodically
copy a top-quartile member's weights + hyperparameters and perturb
`lr`/`gamma` ×0.8/×1.2.

Every mode writes `history.json` to its save dir; compare runs with:
```bash
python plot_compare.py runs/single runs/shared runs/pbt --out docs/plots \
    --wandb-project madmario-eval   # optional W&B mirror
```

### Evaluate a checkpoint
```bash
python train.py evaluate checkpoints/.../mario_net_1.chkpt --n-episodes 20
```

### Record gameplay videos
```bash
# Greedy playback of a checkpoint → MP4s (raw NES frames, not the 84×84 input)
python train.py record checkpoints/.../mario_net_1.chkpt --out-dir videos

# Per-agent capture:
#   PBT — record each agent's own checkpoint
python train.py record runs/pbt/agent_0/mario_net_0.chkpt --label pbt_agent0
#   shared — one learner policy, replayed at each actor's ladder ε
for eps in 0.4 0.079 0.016 0.003; do
  python train.py record runs/shared/mario_net_0.chkpt --epsilon $eps --label actor
done
```
Files are named `<label>_eps<ε>_ep<N>_r<reward>[_FLAG].mp4`; flag-get runs
are tagged in the filename.

---

## MCP server (Claude Code integration)

The MCP server exposes live training controls as tools that Claude Code can
call directly.

**Add to your project `.claude/settings.json`** (already present in this repo):
```json
{
  "mcpServers": {
    "madmario": {
      "command": "/path/to/python3",
      "args": ["/path/to/MadMario/mcp_server.py"],
      "env": { "PYTHONPATH": "/path/to/MadMario" }
    }
  }
}
```

**Available tools:**

| Tool | What it does |
|------|-------------|
| `get_training_status` | Live episode / step / ε / reward metrics |
| `list_checkpoints` | Enumerate saved `.chkpt` files |
| `evaluate_checkpoint` | Run N greedy episodes, return mean reward + flag-get rate |
| `get_curriculum_status` | Current level and success rate |
| `get_config` | Active hyperparameter config |

Push live metrics from your training loop:
```python
from mcp_server import update_state
update_state(episode=e, step=mario.curr_step, epsilon=mario.exploration_rate)
```

---

## Project structure

```
MadMario/
├── madmario/              the installable package (pip install -e .)
│   ├── config.py          Dataclass config for env / agent / wandb / training
│   ├── environment.py     Gymnasium pipeline factory (SkipFrame→Grayscale→Resize→Stack)
│   ├── agent.py           Mario DQN agent — act / cache / learn / save / load
│   ├── neural.py          MarioNet: (Conv+ReLU)×3 → Flatten → (Linear+ReLU) → Q-values
│   ├── replay.py          CPU-side ReplayBuffer (numpy storage, per-batch GPU transfer)
│   ├── wrappers.py        SkipFrame gymnasium wrapper
│   ├── metrics.py         MetricLogger with optional W&B streaming
│   ├── autonomous.py      CurriculumManager, PlateauDetector, SelfImprovementLoop
│   ├── mcp_server.py      MCP server — 5 Claude-callable tools
│   ├── multi_agent.py     Multi-agent v2: shared actor-learner (Ape-X) + PBT
│   ├── cli.py             Typer CLI (train / evaluate) — console script `madmario`
│   └── plotting.py        Cross-run comparison plots — `madmario-compare`
├── train.py               Shim → madmario.cli  (python train.py train still works)
├── plot_compare.py        Shim → madmario.plotting
├── mcp_server.py          Shim → madmario.mcp_server (keeps MCP configs working)
├── main.py                Legacy shim (delegates to the CLI)
├── docs/                  PLAN · COMPARISON · THEORY · EVALUATION · plots/
├── tests/                 46 pytest tests covering all core modules
├── .github/workflows/     CI: pytest + CLI smoke test on every push/PR
├── pyproject.toml         Packaging, console scripts, pytest settings
└── requirements.txt       Pinned dependencies (Python 3.13+)
```

After `pip install -e .` three console scripts are available: `madmario`
(train/evaluate), `madmario-compare` (plots), `madmario-mcp` (MCP server).

---

## Key metrics logged

| Metric | Description |
|--------|-------------|
| `ep_reward` | Total reward for the episode |
| `ep_length` | Steps in the episode |
| `ep_avg_loss` | Mean TD loss over learning steps in the episode |
| `ep_avg_q` | Mean predicted Q-value over learning steps |
| `moving_avg_reward` | 100-episode rolling mean reward |
| `epsilon` | Current exploration rate |

---

## Bug fixes vs original tutorial

| # | Bug | Fix |
|---|-----|-----|
| 1 | Replay buffer stored CUDA tensors — ~22 GB VRAM → OOM | `ReplayBuffer` stores `float32` numpy; tensors created per batch |
| 2 | `curr_step` not in checkpoint — burnin repeated on resume | Saved/restored in every `.chkpt` |
| 3 | Optimizer state never saved — Adam restarts cold | `optimizer.state_dict()` saved/restored |
| 4 | `sync` + `save` fired before burnin guard | Guards moved after burnin check in `learn()` |
| 5 | `reward` as `DoubleTensor` (float64) — spurious upcast | Changed to `FloatTensor` throughout |
| 6 | `td_estimate` / `td_target` used hardcoded `batch_size` index | `torch.arange(state.shape[0])` |
| 7 | `MarioNet.forward` returned `None` on invalid `model` arg | Raises `ValueError` |
| 8 | `if loss:` dropped `loss=0.0` from metrics | Changed to `if loss is not None:` |
| 9 | `learn()` sampled with buffer < batch size (crash with `--burnin 0`) | Minimum-buffer guard in `learn()` / `train_step()` |

---

## Tests

```bash
pytest                  # 46 tests, ~3 s
pytest -v tests/test_multi_agent.py
```

---

## Resources

- Double Q-learning: Hasselt et al., NIPS 2015 — https://arxiv.org/abs/1509.06461
- Reinforcement Learning (Sutton & Barto) — https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf
- OpenAI Spinning Up — https://spinningup.openai.com
- Deep RL Doesn't Work Yet — https://www.alexirpan.com/2018/02/14/rl-hard.html
