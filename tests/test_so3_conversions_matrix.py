import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PASS_XP, TEST_PRECISIONS, get_xp_kwargs, random_quaternion
from scipy.spatial.transform import Rotation as R

from nanomanifold import SO3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_matrix_conversion_cycle(backend, batch_dims, precision, pass_xp):
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    # Create a random SO3 quaternion
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    # Convert to matrix representation
    matrix = SO3.to_matrix(quat, **xp_kwargs)

    assert matrix.dtype == quat.dtype
    assert matrix.shape[:-2] == quat.shape[:-1]
    assert matrix.shape[-2:] == (3, 3)

    # Convert back to quaternion
    quat_converted = SO3.from_matrix(matrix, **xp_kwargs)

    assert quat_converted.dtype == quat.dtype
    assert quat_converted.shape == quat.shape

    # Convert to numpy arrays and compare
    quat_np = np.array(quat)
    quat_converted_np = np.array(quat_converted)

    # Check quaternion equivalence (q and -q represent the same rotation)
    dot_products = np.sum(quat_np * quat_converted_np, axis=-1)
    assert np.allclose(np.abs(dot_products), 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_matrix_conversion_scipy(backend, batch_dims, pass_xp):
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    # Create a random SO3 quaternion
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=32)

    # Convert to matrix representation using nanomanifold
    matrix = SO3.to_matrix(quat, **xp_kwargs)

    # Convert to matrix representation using scipy
    quat_np = np.array(quat)
    # Convert from [w, x, y, z] to scipy's [x, y, z, w] format
    quat_scipy = np.concatenate([quat_np[..., 1:4], quat_np[..., 0:1]], axis=-1)
    r = R.from_quat(quat_scipy.reshape(-1, 4))
    matrix_scipy = r.as_matrix().reshape(matrix.shape)

    assert matrix.dtype == quat.dtype
    assert matrix.shape == matrix_scipy.shape

    matrix_np = np.array(matrix)
    assert np.allclose(matrix_np, matrix_scipy, atol=ATOL[32])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_matrix_differentiality_torch(batch_dims):
    torch = pytest.importorskip("torch")
    # Use double precision for gradient checking as recommended by PyTorch
    dtype = torch.float64

    # Random quaternion input
    quat = random_quaternion(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)

    # Check gradients of SO3.to_matrix
    def f(q):
        return SO3.to_matrix(q)

    assert torch.autograd.gradcheck(f, (quat,), eps=1e-6, atol=1e-5)

    # Matrix input from random quaternion
    matrix = SO3.to_matrix(quat.detach()).requires_grad_(True)

    # Check gradients of SO3.from_matrix
    def g(m):
        return SO3.from_matrix(m)

    assert torch.autograd.gradcheck(g, (matrix,), eps=1e-6, atol=1e-5)
