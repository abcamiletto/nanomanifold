import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PRECISIONS, random_quaternion
from scipy.spatial.transform import Rotation as R

from nanomanifold import SO3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_axis_angle_conversion_cycle(backend, batch_dims, precision):
    # Create a random SO3 quaternion
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    # Convert to axis-angle representation
    axis_angle = SO3.to_axis_angle(quat)

    assert axis_angle.dtype == quat.dtype
    assert axis_angle.shape[:-1] == quat.shape[:-1]

    # Convert back to quaternion
    quat_converted = SO3.from_axis_angle(axis_angle)

    assert quat_converted.dtype == quat.dtype
    assert quat_converted.shape == quat.shape

    # Convert to numpy arrays and compare
    quat_np = np.array(quat)
    quat_converted_np = np.array(quat_converted)

    if precision >= 32:
        # Check quaternion equivalence (q and -q represent the same rotation)
        dot_products = np.sum(quat_np * quat_converted_np, axis=-1)
        assert np.allclose(np.abs(dot_products), 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_axis_angle_conversion_scipy(backend, batch_dims):
    # Create a random SO3 quaternion
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=32)

    # Convert to axis-angle representation using nanomanifold
    axis_angle = SO3.to_axis_angle(quat)

    # Convert to axis-angle representation using scipy
    quat_np = np.array(quat)
    # Convert from [w, x, y, z] to scipy's [x, y, z, w] format
    quat_scipy = np.concatenate([quat_np[..., 1:4], quat_np[..., 0:1]], axis=-1)
    r = R.from_quat(quat_scipy.reshape(-1, 4))
    axis_angle_scipy = r.as_rotvec().reshape(axis_angle.shape)

    assert axis_angle.dtype == quat.dtype
    assert axis_angle.shape == axis_angle_scipy.shape

    axis_angle_np = np.array(axis_angle)
    assert np.allclose(axis_angle_np, axis_angle_scipy, atol=1e-6)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_axis_angle_differentiality_torch(batch_dims):
    torch = pytest.importorskip("torch")
    # Use double precision for gradient checking as recommended by PyTorch
    dtype = torch.float64

    # Random quaternion input
    quat = random_quaternion(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)

    # Check gradients of SO3.to_axis_angle
    def f(q):
        return SO3.to_axis_angle(q)

    assert torch.autograd.gradcheck(f, (quat,), eps=1e-6, atol=1e-5)

    # Axis-angle input from random quaternion
    axis_angle = SO3.to_axis_angle(quat.detach()).requires_grad_(True)

    # Check gradients of SO3.from_axis_angle
    def g(aa):
        return SO3.from_axis_angle(aa)

    assert torch.autograd.gradcheck(g, (axis_angle,), eps=1e-6, atol=1e-5)
