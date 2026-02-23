import pytest

from nanomanifold import SE3, SO3


def test_so3_from_matrix_gradient_finite_torch_at_identity():
    torch = pytest.importorskip("torch")
    matrix = torch.eye(3, dtype=torch.float64, requires_grad=True)

    quat = SO3.from_matrix(matrix)
    quat.sum().backward()

    assert torch.isfinite(matrix.grad).all()


def test_so3_from_matrix_gradient_finite_torch_near_identity():
    torch = pytest.importorskip("torch")
    matrix = torch.eye(3, dtype=torch.float64)
    matrix[0, 1] = 1e-12
    matrix[1, 0] = -1e-12
    matrix.requires_grad_(True)

    quat = SO3.from_matrix(matrix)
    quat.sum().backward()

    assert torch.isfinite(matrix.grad).all()


def test_se3_from_matrix_gradient_finite_torch_at_identity():
    torch = pytest.importorskip("torch")
    matrix = torch.eye(4, dtype=torch.float64, requires_grad=True)

    se3 = SE3.from_matrix(matrix)
    se3.sum().backward()

    assert torch.isfinite(matrix.grad).all()
