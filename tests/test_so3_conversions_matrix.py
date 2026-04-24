import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PASS_XP, TEST_PRECISIONS, get_xp_kwargs, random_quaternion
from scipy.spatial.transform import Rotation as R

from nanomanifold import SO3
from nanomanifold.common import get_namespace_by_name


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_rotmat_conversion_cycle(backend, batch_dims, precision, pass_xp):
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    # Create a random SO3 quaternion
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    # Convert to rotation-matrix representation
    rotmat = SO3.to_rotmat(quat, **xp_kwargs)

    assert rotmat.dtype == quat.dtype
    assert rotmat.shape[:-2] == quat.shape[:-1]
    assert rotmat.shape[-2:] == (3, 3)

    # Convert back to quaternion
    quat_converted = SO3.from_rotmat(rotmat, **xp_kwargs)

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
def test_rotmat_conversion_scipy(backend, batch_dims, pass_xp):
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    # Create a random SO3 quaternion
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=32)

    # Convert to rotation-matrix representation using nanomanifold
    rotmat = SO3.to_rotmat(quat, **xp_kwargs)

    # Convert to matrix representation using scipy
    quat_np = np.array(quat)
    # Convert from [w, x, y, z] to scipy's [x, y, z, w] format
    quat_scipy = np.concatenate([quat_np[..., 1:4], quat_np[..., 0:1]], axis=-1)
    r = R.from_quat(quat_scipy.reshape(-1, 4))
    matrix_scipy = r.as_matrix().reshape(rotmat.shape)

    assert rotmat.dtype == quat.dtype
    assert rotmat.shape == matrix_scipy.shape

    rotmat_np = np.array(rotmat)
    assert np.allclose(rotmat_np, matrix_scipy, atol=ATOL[32])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_rotmat_differentiality_torch(batch_dims):
    torch = pytest.importorskip("torch")
    # Use double precision for gradient checking as recommended by PyTorch
    dtype = torch.float64

    # Random quaternion input
    quat = random_quaternion(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)

    # Check gradients of SO3.to_rotmat
    def f(q):
        return SO3.to_rotmat(q)

    assert torch.autograd.gradcheck(f, (quat,), eps=1e-6, atol=1e-5)

    # Matrix input from random quaternion
    rotmat = SO3.to_rotmat(quat.detach()).requires_grad_(True)

    # Check gradients of SO3.from_rotmat
    def g(m):
        return SO3.from_rotmat(m)

    assert torch.autograd.gradcheck(g, (rotmat,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
@pytest.mark.parametrize("mode", ["svd", "davenport"])
def test_from_matrix_normalize_projects_to_so3(backend, pass_xp, mode):
    xp = get_namespace_by_name(backend)
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    quat = random_quaternion(batch_dims=(2,), backend=backend, precision=32)
    rotmat = SO3.to_rotmat(quat, **xp_kwargs)

    stretch = np.diag(np.array([1.05, 0.97, 1.02], dtype=np.float32))
    noisy_matrix = xp.asarray(np.matmul(np.array(rotmat), stretch))

    quat_converted = SO3.from_matrix(noisy_matrix, mode=mode, **xp_kwargs)
    sixd_converted = SO3.conversions.from_matrix_to_sixd(noisy_matrix, mode=mode, **xp_kwargs)

    quat_np = np.array(quat)
    quat_converted_np = np.array(quat_converted)
    dots = np.sum(quat_np * quat_converted_np, axis=-1)
    assert np.allclose(np.abs(dots), 1.0, atol=ATOL[32])

    sixd_quat_converted = SO3.from_sixd(sixd_converted, **xp_kwargs)
    sixd_quat_converted_np = np.array(sixd_quat_converted)
    sixd_dots = np.sum(quat_np * sixd_quat_converted_np, axis=-1)
    assert np.allclose(np.abs(sixd_dots), 1.0, atol=ATOL[32])


def test_from_matrix_rejects_unknown_projection_mode():
    with pytest.raises(ValueError, match="Unsupported projection mode"):
        SO3.from_matrix(np.eye(3), mode="bad")
