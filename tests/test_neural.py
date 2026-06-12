import pytest
import torch
from neural import MarioNet


def make_net(out: int = 2) -> MarioNet:
    return MarioNet(input_dim=(4, 84, 84), output_dim=out)


def test_forward_online():
    net = make_net()
    x = torch.zeros(1, 4, 84, 84)
    out = net(x, model="online")
    assert out.shape == (1, 2)


def test_forward_target():
    net = make_net()
    x = torch.zeros(1, 4, 84, 84)
    out = net(x, model="target")
    assert out.shape == (1, 2)


def test_forward_invalid_model_raises():
    net = make_net()
    x = torch.zeros(1, 4, 84, 84)
    with pytest.raises(ValueError, match="online.*target"):
        net(x, model="invalid")


def test_target_frozen():
    net = make_net()
    for p in net.target.parameters():
        assert not p.requires_grad


def test_wrong_input_size_raises():
    with pytest.raises(ValueError):
        MarioNet(input_dim=(4, 64, 64), output_dim=2)


def test_output_dim():
    for n in [2, 5, 10]:
        net = MarioNet(input_dim=(4, 84, 84), output_dim=n)
        x = torch.zeros(1, 4, 84, 84)
        assert net(x, model="online").shape == (1, n)
