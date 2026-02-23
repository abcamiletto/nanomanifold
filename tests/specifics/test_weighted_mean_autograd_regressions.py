import pytest

from nanomanifold import SE3, SO3


def test_so3_weighted_mean_propagates_torch_gradients():
    torch = pytest.importorskip("torch")

    q1 = torch.tensor([0.9238795325, 0.3826834324, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    q2 = torch.tensor([0.9238795325, 0.0, 0.3826834324, 0.0], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([0.2, 0.8], dtype=torch.float64)

    out = SO3.weighted_mean([q1, q2], weights, xp=torch)
    loss = (out * torch.tensor([0.7, -0.2, 0.5, 1.1], dtype=torch.float64)).sum()
    loss.backward()

    assert q1.grad is not None
    assert q2.grad is not None
    assert torch.isfinite(q1.grad).all()
    assert torch.isfinite(q2.grad).all()
    assert torch.max(torch.abs(q1.grad)) > 0
    assert torch.max(torch.abs(q2.grad)) > 0


def test_se3_weighted_mean_propagates_rotation_torch_gradients():
    torch = pytest.importorskip("torch")

    se3_1 = torch.tensor([0.9238795325, 0.3826834324, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    se3_2 = torch.tensor([0.9238795325, 0.0, 0.3826834324, 0.0, 1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([0.2, 0.8], dtype=torch.float64)

    out = SE3.weighted_mean([se3_1, se3_2], weights, xp=torch)
    loss = (out[..., :4] * torch.tensor([0.7, -0.2, 0.5, 1.1], dtype=torch.float64)).sum()
    loss.backward()

    assert torch.isfinite(se3_1.grad[..., :4]).all()
    assert torch.isfinite(se3_2.grad[..., :4]).all()
    assert torch.max(torch.abs(se3_1.grad[..., :4])) > 0
    assert torch.max(torch.abs(se3_2.grad[..., :4])) > 0
