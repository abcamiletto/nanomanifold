import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PRECISIONS, random_quaternion
from scipy.spatial.transform import Rotation as R

from nanomanifold import SO3


def get_dtype(backend, precision):
    """Get appropriate dtype for given backend and precision."""
    if backend == "numpy":
        return getattr(np, f"float{precision}")
    elif backend == "torch":
        torch = pytest.importorskip("torch")
        return getattr(torch, f"float{precision}")
    else:  # jax
        jax = pytest.importorskip("jax")
        return getattr(jax.numpy, f"float{precision}")


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_mean_single_quaternion(backend, batch_dims, precision):
    """Test that mean of a single quaternion returns that quaternion"""
    q = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    result = SO3.mean([q])

    assert result.dtype == q.dtype
    assert result.shape == q.shape

    q_np = np.array(q)
    result_np = np.array(result)

    # Check quaternion equivalence (q and -q represent same rotation)
    dot_product = np.sum(q_np * result_np, axis=-1)
    assert np.allclose(np.abs(dot_product), 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_mean_identical_quaternions(backend, batch_dims, precision):
    """Test that mean of identical quaternions returns that quaternion"""
    q = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    # Create list of identical quaternions
    quaternions = [q, q, q, q]
    result = SO3.mean(quaternions)

    assert result.dtype == q.dtype
    assert result.shape == q.shape

    q_np = np.array(q)
    result_np = np.array(result)

    # Check quaternion equivalence
    dot_product = np.sum(q_np * result_np, axis=-1)
    assert np.allclose(np.abs(dot_product), 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_mean_antipodal_quaternions(backend, batch_dims, precision):
    """Test mean of antipodal quaternions (q and -q)"""
    q = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    # Get appropriate backend namespace
    common = pytest.importorskip("nanomanifold.common")
    common.get_namespace_by_name(backend)

    q_neg = -q
    result = SO3.mean([q, q_neg])

    assert result.dtype == q.dtype
    assert result.shape == q.shape

    q_np = np.array(q)
    result_np = np.array(result)

    # Result should be equivalent to either q or -q
    dot_product = np.sum(q_np * result_np, axis=-1)
    assert np.allclose(np.abs(dot_product), 1.0, atol=ATOL[precision])

    # Result should be unit quaternion
    norm = np.linalg.norm(result_np, axis=-1)
    assert np.allclose(norm, 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_weighted_mean_uniform_weights(backend, batch_dims, precision):
    """Test that weighted_mean with uniform weights equals mean"""
    quaternions = [random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision) for _ in range(4)]

    # Get appropriate backend namespace
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend)
    dtype = quaternions[0].dtype

    # Uniform weights as array
    weights = xp.ones(batch_dims + (4,), dtype=dtype)

    result_weighted = SO3.weighted_mean(quaternions, weights)
    result_mean = SO3.mean(quaternions)

    assert result_weighted.dtype == result_mean.dtype
    assert result_weighted.shape == result_mean.shape

    result_weighted_np = np.array(result_weighted)
    result_mean_np = np.array(result_mean)

    # Check quaternion equivalence
    dot_product = np.sum(result_weighted_np * result_mean_np, axis=-1)
    assert np.allclose(np.abs(dot_product), 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_weighted_mean_single_nonzero_weight(backend, batch_dims, precision):
    """Test weighted_mean with only one nonzero weight"""
    quaternions = [random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision) for _ in range(4)]

    # Get appropriate backend namespace
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend)
    dtype = quaternions[0].dtype

    # Only second quaternion has nonzero weight
    weights = xp.asarray([0.0, 1.0, 0.0, 0.0], dtype=dtype)
    # Broadcast to proper batch shape
    if batch_dims:
        weights = xp.broadcast_to(weights, batch_dims + (4,))

    result = SO3.weighted_mean(quaternions, weights)

    assert result.dtype == quaternions[1].dtype
    assert result.shape == quaternions[1].shape

    q1_np = np.array(quaternions[1])
    result_np = np.array(result)

    # Result should be equivalent to the second quaternion
    dot_product = np.sum(q1_np * result_np, axis=-1)
    assert np.allclose(np.abs(dot_product), 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_weighted_mean_array_weights(backend, precision):
    """Test weighted_mean with array weights for batched inputs"""
    batch_dims = (2, 3)

    # Get appropriate backend namespace
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend)
    dtype = get_dtype(backend, precision)

    quaternions = [random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision) for _ in range(3)]

    # Create array weights with shape batch_dims + (n_quats,) = (2, 3, 3)
    weights = xp.asarray(
        [
            [[1.0, 2.0, 1.0], [0.5, 1.0, 0.5], [2.0, 0.5, 1.5]],  # batch [0, :, :] weights for 3 quats
            [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.5, 0.8, 1.2]],  # batch [1, :, :] weights for 3 quats
        ],
        dtype=dtype,
    )

    result = SO3.weighted_mean(quaternions, weights)

    assert result.dtype == quaternions[0].dtype
    assert result.shape == batch_dims + (4,)

    result_np = np.array(result)

    # Result should be unit quaternions
    norms = np.linalg.norm(result_np, axis=-1)
    assert np.allclose(norms, 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_mean_scipy_comparison(backend):
    """Compare mean results with scipy implementation for special cases"""

    # Get appropriate backend namespace
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend)

    # Test with known rotations
    sqrt2_inv = 1.0 / np.sqrt(2.0)

    # Create quaternions in the appropriate backend
    identity = xp.asarray([1.0, 0.0, 0.0, 0.0])
    q_x90 = xp.asarray([sqrt2_inv, sqrt2_inv, 0.0, 0.0])
    q_y90 = xp.asarray([sqrt2_inv, 0.0, sqrt2_inv, 0.0])
    q_z90 = xp.asarray([sqrt2_inv, 0.0, 0.0, sqrt2_inv])

    quaternions = [identity, q_x90, q_y90, q_z90]
    result = SO3.mean(quaternions)

    # Convert to numpy for scipy
    result_np = np.array(result)

    # Convert to scipy format [x, y, z, w] and compute mean
    quaternions_scipy = []
    for q in quaternions:
        q_np = np.array(q)
        q_scipy = np.concatenate([q_np[1:4], q_np[0:1]], axis=-1)  # [w,x,y,z] -> [x,y,z,w]
        quaternions_scipy.append(q_scipy)

    # Scipy's mean rotation
    rotations = R.from_quat(quaternions_scipy)
    mean_rotation = rotations.mean()
    scipy_result = mean_rotation.as_quat()

    # Convert back to [w, x, y, z] format
    scipy_result = np.concatenate([scipy_result[3:4], scipy_result[0:3]], axis=-1)

    # Check quaternion equivalence (allowing for sign flip)
    dot_product = np.sum(result_np * scipy_result)
    assert np.allclose(np.abs(dot_product), 1.0, atol=1e-6)


def test_weighted_mean_zero_norm_quaternion():
    """Test that weighted_mean handles zero norm quaternion gracefully"""
    zero_quat = np.array([0.0, 0.0, 0.0, 0.0])
    valid_quat = np.array([1.0, 0.0, 0.0, 0.0])
    weights = np.array([1.0, 1.0])

    # This should not crash but may give unexpected results due to normalization
    result = SO3.weighted_mean([zero_quat, valid_quat], weights)
    # The result should still be a unit quaternion due to normalization
    assert np.allclose(np.linalg.norm(result), 1.0, atol=1e-6)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_mean_numerical_stability(backend):
    """Test numerical stability with very small rotations"""
    # Get appropriate backend namespace
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend)

    # Very small rotation angles
    identity = xp.asarray([1.0, 0.0, 0.0, 0.0])
    small_eps = 1e-8

    # Small rotations around different axes
    small_x = xp.asarray([1.0, small_eps, 0.0, 0.0])
    small_y = xp.asarray([1.0, 0.0, small_eps, 0.0])
    small_z = xp.asarray([1.0, 0.0, 0.0, small_eps])

    # Normalize
    small_x = small_x / xp.sqrt(xp.sum(small_x**2))
    small_y = small_y / xp.sqrt(xp.sum(small_y**2))
    small_z = small_z / xp.sqrt(xp.sum(small_z**2))

    quaternions = [identity, small_x, small_y, small_z]
    result = SO3.mean(quaternions)

    # Result should be unit quaternion
    result_np = np.array(result)
    norm = np.linalg.norm(result_np)
    assert np.allclose(norm, 1.0, atol=1e-12)

    # Result should be close to identity for small rotations
    dot_with_identity = np.sum(result_np * np.array([1.0, 0.0, 0.0, 0.0]))
    assert dot_with_identity > 0.99


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_weighted_mean_consistency(backend):
    """Test that repeated weighted_mean calls are consistent"""
    # Get appropriate backend namespace
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend)

    quaternions = [random_quaternion((), backend, 32) for _ in range(5)]
    weights = xp.asarray([0.2, 0.3, 0.1, 0.25, 0.15], dtype=quaternions[0].dtype)

    # Compute mean multiple times
    results = []
    for _ in range(5):
        result = SO3.weighted_mean(quaternions, weights)
        results.append(np.array(result))

    # All results should be identical (deterministic)
    for i in range(1, len(results)):
        dot_product = np.sum(results[0] * results[i])
        assert np.allclose(np.abs(dot_product), 1.0, atol=1e-12)


def test_weighted_mean_specific_case():
    """Test weighted mean for a specific known case"""
    # Identity and 90° rotation around z-axis
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    sqrt2_inv = 1.0 / np.sqrt(2.0)
    q_z90 = np.array([sqrt2_inv, 0.0, 0.0, sqrt2_inv])

    # Equal weights should give result similar to slerp at t=0.5
    weights = np.array([1.0, 1.0])
    result = SO3.weighted_mean([identity, q_z90], weights)
    result_np = np.array(result)

    # Compare with slerp at midpoint
    t_mid = np.array([0.5])
    slerp_result = SO3.slerp(identity, q_z90, t_mid)
    slerp_result_np = np.array(slerp_result)[0]  # Remove the N=1 dimension

    # Should agree up to sign ambiguity of unit quaternions
    dot_product = np.sum(result_np * slerp_result_np)
    assert np.allclose(np.abs(dot_product), 1.0, atol=1e-6)


def test_mean_unit_quaternion_output():
    """Test that mean always produces unit quaternions"""
    # Random quaternions with various magnifications
    base_quats = [
        np.array([1.0, 0.5, -0.3, 0.8]),
        np.array([0.7, -0.2, 0.6, 0.1]),
        np.array([-0.4, 0.9, 0.1, -0.7]),
    ]

    # Normalize them properly
    quaternions = []
    for q in base_quats:
        q_normalized = q / np.linalg.norm(q)
        quaternions.append(q_normalized)

    result = SO3.mean(quaternions)
    result_np = np.array(result)

    # Must be unit quaternion
    norm = np.linalg.norm(result_np)
    assert np.allclose(norm, 1.0, atol=1e-12)


def test_weighted_mean_invariant_to_weight_scaling():
    """Test that weighted_mean is invariant to scaling all weights by same factor"""
    quaternions = [random_quaternion((), "numpy", 32) for _ in range(4)]

    weights1 = np.array([0.1, 0.3, 0.4, 0.2])
    weights2 = np.array([1.0, 3.0, 4.0, 2.0])  # Scaled by 10
    weights3 = np.array([0.05, 0.15, 0.2, 0.1])  # Scaled by 0.5

    result1 = SO3.weighted_mean(quaternions, weights1)
    result2 = SO3.weighted_mean(quaternions, weights2)
    result3 = SO3.weighted_mean(quaternions, weights3)

    result1_np = np.array(result1)
    result2_np = np.array(result2)
    result3_np = np.array(result3)

    # All should be equivalent
    dot12 = np.sum(result1_np * result2_np)
    dot13 = np.sum(result1_np * result3_np)

    assert np.allclose(np.abs(dot12), 1.0, atol=1e-12)
    assert np.allclose(np.abs(dot13), 1.0, atol=1e-12)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_mean_jittability_jax(batch_dims):
    """Test that mean functions are JIT-compatible with JAX"""
    jax = pytest.importorskip("jax")

    # Create test quaternions
    quaternions = [random_quaternion(batch_dims=batch_dims, backend="jax", precision=32) for _ in range(3)]

    # Create weights for weighted_mean test
    num_quats = len(quaternions)
    weights = jax.numpy.ones(batch_dims + (num_quats,), dtype=jax.numpy.float32)

    # Define JIT-compiled functions
    @jax.jit
    def jit_mean(quats):
        return SO3.mean(quats)

    @jax.jit
    def jit_weighted_mean(quats, w):
        return SO3.weighted_mean(quats, w)

    # Test that JIT compilation works
    result_mean = jit_mean(quaternions)
    result_weighted = jit_weighted_mean(quaternions, weights)

    # Verify results are valid quaternions
    assert jax.numpy.allclose(jax.numpy.linalg.norm(result_mean, axis=-1), 1.0, atol=1e-6)
    assert jax.numpy.allclose(jax.numpy.linalg.norm(result_weighted, axis=-1), 1.0, atol=1e-6)
