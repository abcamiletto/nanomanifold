import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PRECISIONS, identity_quaternion, random_points, random_quaternion
from scipy.spatial.transform import Rotation as R

from nanomanifold import SO3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_rotate_points_identity(backend, batch_dims, precision):
    """Test that identity quaternion doesn't change points."""
    # Create identity quaternion [1, 0, 0, 0]
    identity = identity_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    # Generate random points
    points = random_points(batch_dims=batch_dims, n_points=10, backend=backend, precision=precision)

    # Apply rotation
    rotated_points = SO3.rotate_points(identity, points)

    assert rotated_points.dtype == points.dtype
    assert rotated_points.shape == points.shape

    # Convert to numpy arrays and compare
    points_np = np.array(points)
    rotated_points_np = np.array(rotated_points)

    assert np.allclose(points_np, rotated_points_np, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_rotate_points_inverse(backend, batch_dims, precision):
    """Test that rotating by q then by q^(-1) gives identity."""
    # Create random quaternion
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    # Create inverse quaternion using SO3.inverse()
    quat_inv = SO3.inverse(quat)

    # Generate random points
    points = random_points(batch_dims=batch_dims, n_points=10, backend=backend, precision=precision)

    # Rotate by q then by q^(-1)
    rotated_points = SO3.rotate_points(quat, points)
    restored_points = SO3.rotate_points(quat_inv, rotated_points)

    assert restored_points.dtype == points.dtype
    assert restored_points.shape == points.shape

    # Convert to numpy arrays and compare
    points_np = np.array(points)
    restored_points_np = np.array(restored_points)

    assert np.allclose(points_np, restored_points_np, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_rotate_points_scipy(backend, batch_dims):
    """Test against scipy implementation."""
    # Create random quaternion and points
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=32)
    points = random_points(batch_dims=batch_dims, n_points=5, backend=backend, precision=32)

    # Rotate using nanomanifold
    rotated_points = SO3.rotate_points(quat, points)

    # Convert to numpy for scipy
    quat_np = np.array(quat)
    points_np = np.array(points)

    # Convert from [w, x, y, z] to scipy's [x, y, z, w] format
    quat_scipy = np.concatenate([quat_np[..., 1:4], quat_np[..., 0:1]], axis=-1)

    # Reshape for scipy processing
    quat_flat = quat_scipy.reshape(-1, 4)
    points_flat = points_np.reshape(-1, points_np.shape[-2], 3)

    # Apply rotation using scipy
    rotated_scipy = []
    for i in range(len(quat_flat)):
        r = R.from_quat(quat_flat[i : i + 1])
        rotated_scipy.append(r.apply(points_flat[i]))
    rotated_scipy = np.array(rotated_scipy).reshape(rotated_points.shape)

    assert rotated_points.dtype == quat.dtype
    assert rotated_points.shape == rotated_scipy.shape

    rotated_points_np = np.array(rotated_points)
    assert np.allclose(rotated_points_np, rotated_scipy, atol=1e-6)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_rotate_points_differentiability_torch(batch_dims):
    """Test differentiability with respect to quaternion and points."""
    torch = pytest.importorskip("torch")
    # Use double precision for gradient checking as recommended by PyTorch
    dtype = torch.float64

    # Create random quaternion and points with gradients
    quat = random_quaternion(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)
    points = random_points(batch_dims=batch_dims, n_points=3, backend="torch", precision=64).to(dtype).requires_grad_(True)

    # Test gradient with respect to quaternion
    def f_quat(q):
        return SO3.rotate_points(q, points.detach())

    assert torch.autograd.gradcheck(f_quat, (quat,), eps=1e-6, atol=1e-5)

    # Test gradient with respect to points
    def f_points(p):
        return SO3.rotate_points(quat.detach(), p)

    assert torch.autograd.gradcheck(f_points, (points,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_rotate_points_jittability_jax(batch_dims):
    """Test that rotate_points function is JIT-compatible with JAX"""
    jax = pytest.importorskip("jax")

    # Create test quaternion and points
    q = random_quaternion(batch_dims=batch_dims, backend="jax", precision=32)
    points = random_points(batch_dims=batch_dims, n_points=5, backend="jax", precision=32)

    # Define JIT-compiled function
    @jax.jit
    def jit_rotate_points(q, pts):
        return SO3.rotate_points(q, pts)

    # Test that JIT compilation works and compare with non-JIT
    result_jit = jit_rotate_points(q, points)
    result_non_jit = SO3.rotate_points(q, points)

    # Verify results match between JIT and non-JIT
    assert jax.numpy.allclose(result_jit, result_non_jit, atol=1e-6)

    # Verify result has correct shape
    expected_shape = batch_dims + (5, 3)
    assert result_jit.shape == expected_shape


def test_rotate_points_specific_cases():
    """Test specific rotation cases."""
    # Test 90-degree rotation around z-axis
    # Quaternion for 90-degree rotation around z: [cos(π/4), 0, 0, sin(π/4)]
    sqrt2_inv = 1.0 / np.sqrt(2.0)
    quat_z90 = np.array([sqrt2_inv, 0, 0, sqrt2_inv])

    # Point at (1, 0, 0) should rotate to (0, 1, 0)
    points = np.array([[1.0, 0.0, 0.0]])
    expected = np.array([[0.0, 1.0, 0.0]])

    rotated = SO3.rotate_points(quat_z90, points)
    rotated_np = np.array(rotated)

    assert np.allclose(rotated_np, expected, atol=1e-6)

    # Test 180-degree rotation around x-axis
    # Quaternion for 180-degree rotation around x: [0, 1, 0, 0]
    quat_x180 = np.array([0.0, 1.0, 0.0, 0.0])

    # Point at (0, 1, 0) should rotate to (0, -1, 0)
    points = np.array([[0.0, 1.0, 0.0]])
    expected = np.array([[0.0, -1.0, 0.0]])

    rotated = SO3.rotate_points(quat_x180, points)
    rotated_np = np.array(rotated)

    assert np.allclose(rotated_np, expected, atol=1e-6)


def test_rotate_points_broadcasting():
    """Test broadcasting between quaternions and points."""
    # Single quaternion, multiple points
    quat = identity_quaternion(batch_dims=(), backend="numpy", precision=32)
    points = random_points(batch_dims=(), n_points=5, backend="numpy", precision=32)

    rotated = SO3.rotate_points(quat, points)
    assert rotated.shape == points.shape
    assert np.allclose(np.array(rotated), np.array(points), atol=1e-6)

    # Multiple quaternions, single point
    quat_batch = identity_quaternion(batch_dims=(2,), backend="numpy", precision=32)
    point = random_points(batch_dims=(), n_points=1, backend="numpy", precision=32)

    rotated = SO3.rotate_points(quat_batch, point)
    assert rotated.shape == (2, 1, 3)
    expected = np.tile(np.array(point), (2, 1, 1))
    assert np.allclose(np.array(rotated), expected, atol=1e-6)
