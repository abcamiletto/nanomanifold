import pytest

from nanomanifold import SO3


def test_distance_gradient_finite_torch_for_identical_rotations():
    torch = pytest.importorskip("torch")
    q = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64, requires_grad=True)

    dist = SO3.distance(q, q, xp=torch)
    dist.backward()

    assert torch.isfinite(q.grad).all()
    assert torch.allclose(q.grad, torch.zeros_like(q.grad), atol=1e-12)


def test_distance_gradient_finite_torch_for_equivalent_sign_flip():
    torch = pytest.importorskip("torch")
    q1 = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64, requires_grad=True)
    q2 = torch.tensor([-0.5, -0.5, -0.5, -0.5], dtype=torch.float64, requires_grad=True)

    dist = SO3.distance(q1, q2, xp=torch)
    dist.backward()

    assert torch.isfinite(q1.grad).all()
    assert torch.isfinite(q2.grad).all()
    assert torch.allclose(q1.grad, torch.zeros_like(q1.grad), atol=1e-12)
    assert torch.allclose(q2.grad, torch.zeros_like(q2.grad), atol=1e-12)
