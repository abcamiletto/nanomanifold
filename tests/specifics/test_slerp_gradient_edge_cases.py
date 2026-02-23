import pytest

from nanomanifold import SE3, SO3


def test_so3_slerp_gradient_finite_torch_for_antipodal_pi_rotation():
    torch = pytest.importorskip("torch")

    q1 = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    q2 = torch.tensor([0.0, -1.0, 0.0, 0.0], dtype=torch.float64)
    t = torch.tensor([0.5], dtype=torch.float64)

    out = SO3.slerp(q1, q2, t, xp=torch)
    out.sum().backward()

    assert torch.isfinite(q1.grad).all()


def test_se3_slerp_rotation_gradient_finite_torch_for_antipodal_pi_rotation():
    torch = pytest.importorskip("torch")

    se3_1 = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.3, -0.2, 0.1], dtype=torch.float64, requires_grad=True)
    se3_2 = torch.tensor([0.0, -1.0, 0.0, 0.0, -0.7, 0.2, -0.5], dtype=torch.float64)
    t = torch.tensor([0.5], dtype=torch.float64)

    out = SE3.slerp(se3_1, se3_2, t, xp=torch)
    out[..., :4].sum().backward()

    assert torch.isfinite(se3_1.grad[..., :4]).all()
