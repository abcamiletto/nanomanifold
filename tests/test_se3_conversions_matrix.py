import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PASS_XP, TEST_PRECISIONS, get_xp_kwargs, random_se3

from nanomanifold import SE3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_matrix_conversion_cycle(backend, batch_dims, precision, pass_xp):
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    matrix = SE3.to_matrix(se3, **xp_kwargs)

    assert matrix.dtype == se3.dtype
    assert matrix.shape[:-2] == se3.shape[:-1]
    assert matrix.shape[-2:] == (4, 4)

    se3_converted = SE3.from_matrix(matrix, **xp_kwargs)

    assert se3_converted.dtype == se3.dtype
    assert se3_converted.shape == se3.shape

    se3_np = np.array(se3)
    se3_converted_np = np.array(se3_converted)

    # Check quaternion equivalence (q and -q represent the same rotation)
    quat_orig = se3_np[..., :4]
    quat_conv = se3_converted_np[..., :4]
    dot_products = np.sum(quat_orig * quat_conv, axis=-1)
    assert np.allclose(np.abs(dot_products), 1.0, atol=ATOL[precision])

    trans_orig = se3_np[..., 4:7]
    trans_conv = se3_converted_np[..., 4:7]
    assert np.allclose(trans_orig, trans_conv, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_matrix_properties(backend, batch_dims, pass_xp):
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=32)

    matrix = SE3.to_matrix(se3, **xp_kwargs)
    matrix_np = np.array(matrix)

    expected_bottom_row = np.zeros(matrix_np.shape[:-2] + (4,))
    expected_bottom_row[..., 3] = 1
    assert np.allclose(matrix_np[..., 3, :], expected_bottom_row, atol=1e-6)

    R = matrix_np[..., :3, :3]
    RTR = np.matmul(np.transpose(R, axes=list(range(R.ndim - 2)) + [-1, -2]), R)
    I = np.eye(3)
    I_batched = np.broadcast_to(I, R.shape[:-2] + (3, 3))
    assert np.allclose(RTR, I_batched, atol=1e-6)

    det = np.linalg.det(R)
    assert np.allclose(det, 1.0, atol=1e-6)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_matrix_differentiality_torch(batch_dims):
    torch = pytest.importorskip("torch")
    dtype = torch.float64

    se3 = random_se3(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)

    def f(s):
        return SE3.to_matrix(s)

    assert torch.autograd.gradcheck(f, (se3,), eps=1e-6, atol=1e-5)

    matrix = SE3.to_matrix(se3.detach()).requires_grad_(True)

    def g(m):
        return SE3.from_matrix(m)

    assert torch.autograd.gradcheck(g, (matrix,), eps=1e-6, atol=1e-5)
