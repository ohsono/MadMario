import math
import tempfile
from pathlib import Path

import pytest
from metrics import MetricLogger


def make_logger(tmp_path: Path) -> MetricLogger:
    return MetricLogger(save_dir=tmp_path, use_wandb=False)


def test_log_step_accumulates_reward(tmp_path):
    logger = make_logger(tmp_path)
    logger.log_step(reward=10.0, loss=None, q=None)
    logger.log_step(reward=5.0, loss=None, q=None)
    assert logger.curr_ep_reward == 15.0


def test_log_step_none_loss_not_counted(tmp_path):
    logger = make_logger(tmp_path)
    logger.log_step(reward=1.0, loss=None, q=None)
    assert logger.curr_ep_loss_length == 0


def test_log_step_zero_loss_is_counted(tmp_path):
    """Regression: `if loss:` dropped loss=0.0; must use `if loss is not None:`"""
    logger = make_logger(tmp_path)
    logger.log_step(reward=1.0, loss=0.0, q=0.0)
    assert logger.curr_ep_loss_length == 1
    assert logger.curr_ep_loss == 0.0
    assert logger.curr_ep_q == 0.0


def test_log_step_positive_loss_counted(tmp_path):
    logger = make_logger(tmp_path)
    logger.log_step(reward=1.0, loss=0.5, q=0.3)
    assert logger.curr_ep_loss_length == 1
    assert math.isclose(logger.curr_ep_loss, 0.5)


def test_log_episode_resets(tmp_path):
    logger = make_logger(tmp_path)
    logger.log_step(reward=5.0, loss=1.0, q=0.5)
    logger.log_episode()
    assert logger.curr_ep_reward == 0.0
    assert logger.curr_ep_loss_length == 0
    assert len(logger.ep_rewards) == 1


def test_log_episode_avg_loss_zero_when_no_learning(tmp_path):
    logger = make_logger(tmp_path)
    logger.log_step(reward=1.0, loss=None, q=None)
    logger.log_episode()
    assert logger.ep_avg_losses[0] == 0.0


def test_record_writes_log_file(tmp_path):
    logger = make_logger(tmp_path)
    for _ in range(5):
        logger.log_step(reward=1.0, loss=0.1, q=0.2)
        logger.log_episode()
    logger.record(episode=5, epsilon=0.9, step=100)
    log_content = (tmp_path / "log").read_text()
    assert "5" in log_content
