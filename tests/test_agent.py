"""Agent unit tests — exercise act/cache/learn/save/load without a real environment."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from madmario.config import AgentConfig
from madmario.agent import Mario

STATE_DIM = (4, 84, 84)
ACTION_DIM = 2
DEVICE = torch.device("cpu")


def make_agent(tmp_path: Path, burnin: int = 0) -> Mario:
    cfg = AgentConfig(
        batch_size=8,
        burnin=burnin,
        learn_every=1,
        sync_every=1_000_000,
        save_every=1_000_000,
        memory_size=500,
        exploration_rate=1.0,
    )
    return Mario(STATE_DIM, ACTION_DIM, tmp_path, cfg)


def random_state() -> np.ndarray:
    return np.random.rand(*STATE_DIM).astype(np.float32)


# ------------------------------------------------------------------
# act
# ------------------------------------------------------------------

def test_act_returns_valid_action(tmp_path):
    mario = make_agent(tmp_path)
    for _ in range(20):
        a = mario.act(random_state())
        assert 0 <= a < ACTION_DIM


def test_act_increments_step(tmp_path):
    mario = make_agent(tmp_path)
    assert mario.curr_step == 0
    mario.act(random_state())
    assert mario.curr_step == 1


def test_exploration_rate_decays(tmp_path):
    mario = make_agent(tmp_path)
    eps_before = mario.exploration_rate
    mario.act(random_state())
    assert mario.exploration_rate <= eps_before


# ------------------------------------------------------------------
# cache
# ------------------------------------------------------------------

def test_cache_fills_buffer(tmp_path):
    mario = make_agent(tmp_path)
    for _ in range(10):
        mario.cache(random_state(), random_state(), 0, 1.0, False)
    assert len(mario.memory) == 10


def test_cache_respects_capacity(tmp_path):
    cfg = AgentConfig(memory_size=5, batch_size=2, burnin=0)
    mario = Mario(STATE_DIM, ACTION_DIM, tmp_path, cfg)
    for _ in range(20):
        mario.cache(random_state(), random_state(), 0, 1.0, False)
    assert len(mario.memory) == 5


# ------------------------------------------------------------------
# learn
# ------------------------------------------------------------------

def test_learn_returns_none_during_burnin(tmp_path):
    mario = make_agent(tmp_path, burnin=10_000)
    mario.cache(random_state(), random_state(), 0, 1.0, False)
    q, loss = mario.learn()
    assert q is None and loss is None


def test_learn_returns_floats_after_burnin(tmp_path):
    mario = make_agent(tmp_path, burnin=0)
    for _ in range(20):
        mario.cache(random_state(), random_state(), 0, 1.0, False)
        mario.act(random_state())  # increment curr_step
    q, loss = mario.learn()
    assert q is not None and loss is not None
    assert isinstance(q, float) and isinstance(loss, float)


# ------------------------------------------------------------------
# save / load (FIX: curr_step + optimizer preserved)
# ------------------------------------------------------------------

def test_save_creates_file(tmp_path):
    mario = make_agent(tmp_path)
    mario.curr_step = 500_000
    mario.cfg.save_every = 500_000
    path = mario.save()
    assert path.exists()


def test_save_includes_curr_step(tmp_path):
    mario = make_agent(tmp_path)
    mario.curr_step = 12345
    mario.cfg.save_every = 12345
    path = mario.save()
    ckp = torch.load(path, map_location="cpu", weights_only=False)
    assert ckp["curr_step"] == 12345


def test_save_includes_optimizer(tmp_path):
    mario = make_agent(tmp_path)
    mario.curr_step = 500_000
    mario.cfg.save_every = 500_000
    path = mario.save()
    ckp = torch.load(path, map_location="cpu", weights_only=False)
    assert "optimizer" in ckp


def test_load_restores_curr_step(tmp_path):
    mario = make_agent(tmp_path)
    mario.curr_step = 99_999
    mario.cfg.save_every = 99_999
    path = mario.save()

    mario2 = make_agent(tmp_path)
    mario2.load(path)
    assert mario2.curr_step == 99_999


def test_load_restores_exploration_rate(tmp_path):
    mario = make_agent(tmp_path)
    mario.exploration_rate = 0.42
    mario.curr_step = 500_000
    mario.cfg.save_every = 500_000
    path = mario.save()

    mario2 = make_agent(tmp_path)
    mario2.load(path)
    assert abs(mario2.exploration_rate - 0.42) < 1e-6


def test_load_missing_file_raises(tmp_path):
    mario = make_agent(tmp_path)
    with pytest.raises(FileNotFoundError):
        mario.load(tmp_path / "nonexistent.chkpt")


def test_save_to_explicit_path(tmp_path):
    mario = make_agent(tmp_path)
    target = tmp_path / "best.chkpt"
    out = mario.save(target)
    assert out == target and target.exists()
