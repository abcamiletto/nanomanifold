import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PRECISIONS, identity_quaternion, random_quaternion
from scipy.spatial.transform import Rotation as R

from nanomanifold import SO3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_distance_identity(backend, batch_dims, precision):
    # Distance from any quaternion to itself should be 0
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)
    distance = SO3.distance(quat, quat)

    assert distance.dtype == quat.dtype
    assert distance.shape == batch_dims

    distance_np = np.array(distance)

    # Allow appropriate tolerance for numerical precision
    assert np.allclose(distance_np, 0.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_distance_identity_quaternion(backend, batch_dims, precision):
    # Distance between any quaternion and identity should equal distance to canonical identity
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)
    identity = identity_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    distance = SO3.distance(quat, identity)

    assert distance.dtype == quat.dtype
    assert distance.shape == batch_dims

    distance_np = np.array(distance)

    # Distance should be in [0, π]
    tol = ATOL[precision]
    assert np.all(distance_np >= -tol)
    assert np.all(distance_np <= np.pi + tol)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_distance_symmetry(backend, batch_dims, precision):
    # Distance should be symmetric: d(q1, q2) = d(q2, q1)
    q1 = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)
    q2 = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    dist1 = SO3.distance(q1, q2)
    dist2 = SO3.distance(q2, q1)

    assert dist1.dtype == q1.dtype
    assert dist1.shape == batch_dims
    assert dist2.dtype == q1.dtype
    assert dist2.shape == batch_dims

    dist1_np = np.array(dist1)
    dist2_np = np.array(dist2)
    assert np.allclose(dist1_np, dist2_np, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_distance_quaternion_double_cover(backend, batch_dims, precision):
    # Distance between q and -q should be 0 (same rotation)
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)
    quat_neg = -quat

    distance = SO3.distance(quat, quat_neg)

    assert distance.dtype == quat.dtype
    assert distance.shape == batch_dims

    distance_np = np.array(distance)
    # Allow appropriate tolerance for numerical precision
    assert np.allclose(distance_np, 0.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_distance_scipy(backend, batch_dims):
    # Compare with scipy's angle calculation
    q1 = random_quaternion(batch_dims=batch_dims, backend=backend, precision=32)
    q2 = random_quaternion(batch_dims=batch_dims, backend=backend, precision=32)

    # Compute distance using nanomanifold
    distance = SO3.distance(q1, q2)

    # Convert to numpy for scipy
    q1_np = np.array(q1)
    q2_np = np.array(q2)

    # Convert from [w, x, y, z] to scipy's [x, y, z, w] format
    q1_scipy = np.concatenate([q1_np[..., 1:4], q1_np[..., 0:1]], axis=-1)
    q2_scipy = np.concatenate([q2_np[..., 1:4], q2_np[..., 0:1]], axis=-1)

    # Reshape for scipy processing
    q1_flat = q1_scipy.reshape(-1, 4)
    q2_flat = q2_scipy.reshape(-1, 4)

    # Compute angles using scipy
    scipy_angles = []
    for i in range(len(q1_flat)):
        r1 = R.from_quat(q1_flat[i : i + 1])
        r2 = R.from_quat(q2_flat[i : i + 1])
        # Compute relative rotation
        r_rel = r1.inv() * r2
        # Get magnitude of rotation angle
        angle = np.abs(r_rel.magnitude())
        scipy_angles.append(angle)

    scipy_angles = np.array(scipy_angles).reshape(batch_dims)

    assert distance.dtype == q1.dtype
    assert distance.shape == scipy_angles.shape

    distance_np = np.array(distance)
    assert np.allclose(distance_np, scipy_angles, atol=1e-6)


def test_distance_specific_rotations():
    # Test distance between specific known rotations
    identity = np.array([1.0, 0.0, 0.0, 0.0])

    # 90-degree rotation around z-axis: angle = π/2
    sqrt2_inv = 1.0 / np.sqrt(2.0)
    q_z90 = np.array([sqrt2_inv, 0, 0, sqrt2_inv])

    # 180-degree rotation around z-axis: angle = π
    q_z180 = np.array([0.0, 0.0, 0.0, 1.0])

    # Distance from identity to 90° rotation should be π/2
    dist_90 = SO3.distance(identity, q_z90)
    dist_90_np = np.array(dist_90)
    assert np.allclose(dist_90_np, np.pi / 2, atol=1e-6)

    # Distance from identity to 180° rotation should be π
    dist_180 = SO3.distance(identity, q_z180)
    dist_180_np = np.array(dist_180)
    assert np.allclose(dist_180_np, np.pi, atol=1e-6)

    # Distance from 90° to 180° should be π/2
    dist_90_to_180 = SO3.distance(q_z90, q_z180)
    dist_90_to_180_np = np.array(dist_90_to_180)
    assert np.allclose(dist_90_to_180_np, np.pi / 2, atol=1e-6)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_distance_differentiability_torch(batch_dims):
    torch = pytest.importorskip("torch")
    # Use double precision for gradient checking
    dtype = torch.float64

    # Random quaternion inputs
    q1 = random_quaternion(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)
    q2 = random_quaternion(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)

    # Check gradients with respect to first argument
    def f_q1(q):
        return SO3.distance(q, q2.detach()).sum()

    assert torch.autograd.gradcheck(f_q1, (q1,), eps=1e-6, atol=1e-5)

    # Check gradients with respect to second argument
    def f_q2(q):
        return SO3.distance(q1.detach(), q).sum()

    assert torch.autograd.gradcheck(f_q2, (q2,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_distance_jittability_jax(batch_dims):
    """Test that distance function is JIT-compatible with JAX"""
    jax = pytest.importorskip("jax")

    # Create test quaternions
    q1 = random_quaternion(batch_dims=batch_dims, backend="jax", precision=32)
    q2 = random_quaternion(batch_dims=batch_dims, backend="jax", precision=32)

    # Define JIT-compiled function
    @jax.jit
    def jit_distance(q1, q2):
        return SO3.distance(q1, q2)

    # Test that JIT compilation works and compare with non-JIT
    result_jit = jit_distance(q1, q2)
    result_non_jit = SO3.distance(q1, q2)

    # Verify results match between JIT and non-JIT
    assert jax.numpy.allclose(result_jit, result_non_jit, atol=1e-6)

    # Verify result has correct shape and is non-negative
    assert result_jit.shape == batch_dims
    assert jax.numpy.all(result_jit >= 0.0)


def test_distance_range():
    # Distance should always be in range [0, π]
    q1 = random_quaternion(batch_dims=(100,), backend="numpy", precision=32)
    q2 = random_quaternion(batch_dims=(100,), backend="numpy", precision=32)

    distances = SO3.distance(q1, q2)
    distances_np = np.array(distances)

    assert np.all(distances_np >= 0.0)
    assert np.all(distances_np <= np.pi + 1e-6)


def test_distance_triangle_inequality():
    # Triangle inequality: d(q1,q3) <= d(q1,q2) + d(q2,q3)
    q1 = random_quaternion(batch_dims=(10,), backend="numpy", precision=32)
    q2 = random_quaternion(batch_dims=(10,), backend="numpy", precision=32)
    q3 = random_quaternion(batch_dims=(10,), backend="numpy", precision=32)

    d12 = np.array(SO3.distance(q1, q2))
    d23 = np.array(SO3.distance(q2, q3))
    d13 = np.array(SO3.distance(q1, q3))

    # Triangle inequality should hold (with small tolerance for numerical errors)
    assert np.all(d13 <= d12 + d23 + 1e-6)


def test_distance_broadcasting():
    # Test broadcasting between different batch sizes
    q_single = random_quaternion(batch_dims=(), backend="numpy", precision=32)
    q_batch = random_quaternion(batch_dims=(3,), backend="numpy", precision=32)

    # Test broadcasting
    distance = SO3.distance(q_single, q_batch)

    assert distance.shape == (3,)

    distance_np = np.array(distance)
    assert np.all(distance_np >= 0.0)
    assert np.all(distance_np <= np.pi + 1e-6)
