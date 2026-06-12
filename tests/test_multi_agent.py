"""Tests for multi-agent v2: epsilon ladder, PBT helpers, learner train_step."""
import random
from pathlib import Path

import numpy as np
import pytest

from madmario.config import AgentConfig
from madmario.agent import Mario
from madmario.multi_agent import epsilon_ladder, perturb_hparams, pbt_partition


# ---------------------------------------------------------------------------
# Epsilon ladder (Ape-X)
# ---------------------------------------------------------------------------

def test_epsilon_ladder_endpoints():
    n, base, alpha = 4, 0.4, 1 / 128
    assert epsilon_ladder(0, n, base, alpha) == pytest.approx(base)
    assert epsilon_ladder(n - 1, n, base, alpha) == pytest.approx(base * alpha)


def test_epsilon_ladder_monotonic_decreasing():
    eps = [epsilon_ladder(i, 8) for i in range(8)]
    assert all(a > b for a, b in zip(eps, eps[1:]))


def test_epsilon_ladder_single_actor():
    assert epsilon_ladder(0, 1, base=0.4) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# PBT explore / exploit helpers
# ---------------------------------------------------------------------------

def test_perturb_hparams_multiplies_by_08_or_12():
    rng = random.Random(0)
    h = {"lr": 0.00025, "gamma": 0.9}
    out = perturb_hparams(h, rng)
    assert out["lr"] in (pytest.approx(0.0002), pytest.approx(0.0003))
    assert out["gamma"] in (pytest.approx(0.72), pytest.approx(0.997))


def test_perturb_hparams_clamps_gamma():
    rng = random.Random(1)
    for _ in range(20):
        out = perturb_hparams({"lr": 0.001, "gamma": 0.99}, rng)
        assert 0.5 <= out["gamma"] <= 0.997
        assert 1e-6 <= out["lr"] <= 1e-2


def test_perturb_does_not_mutate_input():
    h = {"lr": 0.001, "gamma": 0.9}
    perturb_hparams(h, random.Random(2))
    assert h == {"lr": 0.001, "gamma": 0.9}


def test_pbt_partition_ranks_current_means():
    means = {0: 10.0, 1: 50.0, 2: 30.0, 3: 90.0}
    bottom, top = pbt_partition(means, 0.25)
    assert bottom == [0]
    assert top == [3]


def test_pbt_partition_small_population():
    bottom, top = pbt_partition({0: 1.0, 1: 2.0}, 0.25)
    assert bottom == [0] and top == [1]  # k floors to at least 1


# ---------------------------------------------------------------------------
# Learner-side train_step (shared mode)
# ---------------------------------------------------------------------------

@pytest.fixture
def small_mario(tmp_path: Path) -> Mario:
    cfg = AgentConfig(
        batch_size=4, burnin=0, learn_every=1, lr=0.0,
        sync_every=8, save_every=10_000, memory_size=64,
    )
    return Mario(state_dim=(4, 84, 84), action_dim=2, save_dir=tmp_path, config=cfg)


def _fill(mario: Mario, n: int) -> None:
    s = np.zeros((4, 84, 84), dtype=np.float32)
    for i in range(n):
        mario.cache(s, s, i % 2, 1.0, False)


def test_train_step_skips_when_buffer_below_batch(small_mario):
    _fill(small_mario, 2)  # batch_size is 4
    q, loss = small_mario.train_step()
    assert q is None and loss is None
    assert small_mario.learn_step_count == 0


def test_train_step_learns_and_counts(small_mario):
    _fill(small_mario, 8)
    q, loss = small_mario.train_step()
    assert isinstance(q, float) and isinstance(loss, float)
    assert small_mario.learn_step_count == 1


def test_train_step_syncs_target_at_period(small_mario):
    _fill(small_mario, 16)
    # sync_period = sync_every // learn_every = 8 gradient steps
    for _ in range(7):
        small_mario.train_step()
    small_mario.net.online[0].weight.data.add_(1.0)  # force divergence
    small_mario.train_step()  # 8th step -> sync
    online = small_mario.net.online.state_dict()
    target = small_mario.net.target.state_dict()
    for k in online:
        assert (online[k] == target[k]).all()
