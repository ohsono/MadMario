import numpy as np
import torch
import pytest
from madmario.replay import ReplayBuffer


DEVICE = torch.device("cpu")
STATE_SHAPE = (4, 84, 84)


def _push_n(buf: ReplayBuffer, n: int) -> None:
    for _ in range(n):
        s = np.random.rand(*STATE_SHAPE).astype(np.float32)
        ns = np.random.rand(*STATE_SHAPE).astype(np.float32)
        buf.push(s, ns, action=0, reward=1.0, done=False)


def test_len():
    buf = ReplayBuffer(capacity=100)
    _push_n(buf, 10)
    assert len(buf) == 10


def test_capacity_cap():
    buf = ReplayBuffer(capacity=5)
    _push_n(buf, 10)
    assert len(buf) == 5


def test_sample_shapes():
    buf = ReplayBuffer(capacity=100)
    _push_n(buf, 50)
    s, ns, a, r, d = buf.sample(batch_size=32, device=DEVICE)
    assert s.shape == (32, *STATE_SHAPE)
    assert ns.shape == (32, *STATE_SHAPE)
    assert a.shape == (32,)
    assert r.shape == (32,)
    assert d.shape == (32,)


def test_sample_dtypes():
    buf = ReplayBuffer(capacity=100)
    _push_n(buf, 50)
    s, ns, a, r, d = buf.sample(32, DEVICE)
    assert s.dtype == torch.float32
    assert ns.dtype == torch.float32
    assert a.dtype == torch.long
    assert r.dtype == torch.float32      # FIX: was DoubleTensor before
    assert d.dtype == torch.bool


def test_sample_on_device():
    buf = ReplayBuffer(capacity=100)
    _push_n(buf, 50)
    s, ns, a, r, d = buf.sample(32, DEVICE)
    for t in (s, ns, a, r, d):
        assert t.device.type == "cpu"


def test_sample_too_large_raises():
    buf = ReplayBuffer(capacity=100)
    _push_n(buf, 10)
    with pytest.raises(ValueError):
        buf.sample(20, DEVICE)
