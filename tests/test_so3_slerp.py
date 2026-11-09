import numpy as np
import pytest
from conftest import ATOL, TEST_BATCH_DIMS, TEST_PRECISIONS, random_quaternion
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from nanomanifold import SO3


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_slerp_endpoints(batch_dims, precision):
    """Test that slerp(q1, q2, [0]) = q1 and slerp(q1, q2, [1]) = q2"""
    q1 = random_quaternion(batch_dims=batch_dims, precision=precision)
    q2 = random_quaternion(batch_dims=batch_dims, precision=precision)

    # Get appropriate backend namespace
    xp = np
    dtype = np.dtype(f"float{precision}")

    # Test t=[0] case
    t_0 = xp.asarray([0.0], dtype=dtype)
    result_0 = SO3.slerp(q1, q2, t_0)

    # Test t=[1] case
    t_1 = xp.asarray([1.0], dtype=dtype)
    result_1 = SO3.slerp(q1, q2, t_1)

    assert result_0.dtype == q1.dtype
    assert result_0.shape == batch_dims + (1, 4)
    assert result_1.dtype == q1.dtype
    assert result_1.shape == batch_dims + (1, 4)

    q1_np = np.array(q1)
    q2_np = np.array(q2)
    result_0_np = np.array(result_0)
    result_1_np = np.array(result_1)

    # Remove the N=1 dimension for comparison
    result_0_squeezed = result_0_np.squeeze(-2)  # [..., 1, 4] -> [..., 4]
    result_1_squeezed = result_1_np.squeeze(-2)  # [..., 1, 4] -> [..., 4]

    # Check quaternion equivalence (q and -q represent same rotation)
    dot_products_0 = np.sum(q1_np * result_0_squeezed, axis=-1)
    dot_products_1 = np.sum(q2_np * result_1_squeezed, axis=-1)
    assert np.allclose(np.abs(dot_products_0), 1.0, atol=ATOL[precision])
    assert np.allclose(np.abs(dot_products_1), 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_slerp_multiple_points(batch_dims, precision):
    """Test slerp with multiple interpolation points"""
    q1 = random_quaternion(batch_dims=batch_dims, precision=precision)
    q2 = random_quaternion(batch_dims=batch_dims, precision=precision)

    # Get appropriate backend namespace
    xp = np
    dtype = np.dtype(f"float{precision}")

    # Test multiple interpolation points
    t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    t_array = xp.asarray(t_values, dtype=dtype)

    result = SO3.slerp(q1, q2, t_array)

    assert result.dtype == q1.dtype
    assert result.shape == batch_dims + (5, 4)

    # Results should be unit quaternions
    result_np = np.array(result)
    norms = np.linalg.norm(result_np, axis=-1)
    assert np.allclose(norms, 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_slerp_same_quaternion(batch_dims, precision):
    """Test that slerp(q, q, t) = q for any t"""
    q = random_quaternion(batch_dims=batch_dims, precision=precision)

    # Get appropriate backend namespace
    xp = np
    dtype = np.dtype(f"float{precision}")

    # Test several interpolation values
    t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    t_array = xp.asarray(t_values, dtype=dtype)

    result = SO3.slerp(q, q, t_array)

    assert result.dtype == q.dtype
    assert result.shape == batch_dims + (5, 4)

    q_np = np.array(q)

    result_np = np.array(result)

    # Expand q for comparison with result
    q_expanded = np.expand_dims(q_np, axis=-2)  # [..., 1, 4]
    q_broadcasted = np.broadcast_to(q_expanded, result_np.shape)  # [..., 5, 4]

    # Check quaternion equivalence
    dot_products = np.sum(q_broadcasted * result_np, axis=-1)
    assert np.allclose(np.abs(dot_products), 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_slerp_scipy_comparison(batch_dims):
    """Compare slerp results with scipy implementation"""
    q1 = random_quaternion(batch_dims=batch_dims, precision=32)
    q2 = random_quaternion(batch_dims=batch_dims, precision=32)

    # Get appropriate backend namespace
    xp = np
    dtype = np.float32

    # Test several interpolation values
    t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    t_array = xp.asarray(t_values, dtype=dtype)

    result = SO3.slerp(q1, q2, t_array)

    # Convert to numpy for scipy
    q1_np = np.array(q1)
    q2_np = np.array(q2)

    # Convert from [w, x, y, z] to scipy's [x, y, z, w] format
    q1_scipy = np.concatenate([q1_np[..., 1:4], q1_np[..., 0:1]], axis=-1)
    q2_scipy = np.concatenate([q2_np[..., 1:4], q2_np[..., 0:1]], axis=-1)

    # Reshape for scipy processing
    original_shape = q1_scipy.shape
    q1_flat = q1_scipy.reshape(-1, 4)
    q2_flat = q2_scipy.reshape(-1, 4)

    # Process each quaternion pair
    scipy_results = []
    for i in range(len(q1_flat)):
        r1 = R.from_quat(q1_flat[i : i + 1])
        r2 = R.from_quat(q2_flat[i : i + 1])

        # Create slerp interpolator and get all t values at once
        slerp = Slerp([0.0, 1.0], R.concatenate([r1, r2]))
        r_interp = slerp(t_values)
        scipy_results.append(r_interp.as_quat())

    scipy_result = np.array(scipy_results).reshape(original_shape[:-1] + (5, 4))
    # Convert back to [w, x, y, z] format
    scipy_result = np.concatenate([scipy_result[..., 3:4], scipy_result[..., 0:3]], axis=-1)

    assert result.shape == scipy_result.shape

    result_np = np.array(result)
    # Check quaternion equivalence (q and -q represent same rotation)
    dot_products = np.sum(result_np * scipy_result, axis=-1)
    assert np.allclose(np.abs(dot_products), 1.0, atol=1e-6)


def test_slerp_specific_rotations():
    """Test slerp for known rotations"""
    # Identity quaternion
    identity = np.array([1.0, 0.0, 0.0, 0.0])

    # 90-degree rotation around z-axis
    sqrt2_inv = 1.0 / np.sqrt(2.0)
    q_z90 = np.array([sqrt2_inv, 0.0, 0.0, sqrt2_inv])

    # Test interpolation from identity to 90° rotation at t=0.5
    t_mid = np.array([0.5])
    result_mid = SO3.slerp(identity, q_z90, t_mid)
    result_mid_np = np.array(result_mid)

    # This should give us a 45° rotation around z-axis
    cos45 = np.cos(np.pi / 8)  # cos(45°/2)
    sin45 = np.sin(np.pi / 8)  # sin(45°/2)
    expected_mid = np.array([cos45, 0.0, 0.0, sin45])

    # Check quaternion equivalence
    dot_product = np.sum(result_mid_np[0] * expected_mid)
    assert np.allclose(np.abs(dot_product), 1.0, atol=1e-6)


def test_slerp_broadcasting():
    """Test broadcasting between different batch sizes and t dimensions"""
    # Single quaternion pair with multiple t values
    q1_single = np.array([1.0, 0.0, 0.0, 0.0])
    sqrt2_inv = 1.0 / np.sqrt(2.0)
    q2_single = np.array([sqrt2_inv, 0.0, 0.0, sqrt2_inv])

    # Multiple t values
    t_multiple = np.array([0.0, 0.5, 1.0])
    result = SO3.slerp(q1_single, q2_single, t_multiple)

    assert result.shape == (3, 4)

    # Results should be unit quaternions
    result_np = np.array(result)
    norms = np.linalg.norm(result_np, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-6)

    # Batch quaternions with single t
    q1_batch = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])  # 2 identities
    q2_batch = np.array([[sqrt2_inv, 0.0, 0.0, sqrt2_inv], [sqrt2_inv, sqrt2_inv, 0.0, 0.0]])  # z90, x90

    t_single = np.array([0.5])
    result_batch = SO3.slerp(q1_batch, q2_batch, t_single)

    assert result_batch.shape == (2, 1, 4)

    # Results should be unit quaternions
    result_batch_np = np.array(result_batch)
    norms_batch = np.linalg.norm(result_batch_np, axis=-1)
    assert np.allclose(norms_batch, 1.0, atol=1e-6)


def test_slerp_antipodal_quaternions():
    """Test slerp behavior with antipodal quaternions (q and -q)"""
    q = np.array([0.6, 0.8, 0.0, 0.0])  # Some quaternion
    q = q / np.linalg.norm(q)  # Normalize
    q_neg = -q  # Antipodal quaternion (represents same rotation)

    # Interpolation between q and -q should take the shorter path
    t_mid = np.array([0.5])
    result = SO3.slerp(q, q_neg, t_mid)
    result_np = np.array(result)

    # Result should be unit quaternion
    norm = np.linalg.norm(result_np)
    assert np.allclose(norm, 1.0, atol=1e-6)

    # The midpoint should be equivalent to q (or -q) since they're the same rotation
    dot_product = np.sum(result_np[0] * q)
    assert np.allclose(np.abs(dot_product), 1.0, atol=1e-6)


def test_slerp_edge_cases():
    """Test edge cases and numerical stability"""
    # Very close quaternions
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([1.0 - 1e-10, 1e-10, 0.0, 0.0])
    q2 = q2 / np.linalg.norm(q2)  # Normalize

    t_mid = np.array([0.5])
    result = SO3.slerp(q1, q2, t_mid)
    result_np = np.array(result)

    # Result should be unit quaternion
    norm = np.linalg.norm(result_np)
    assert np.allclose(norm, 1.0, atol=1e-12)

    # Result should be between q1 and q2
    dot1 = np.sum(result_np[0] * q1)
    dot2 = np.sum(result_np[0] * q2)
    assert dot1 > 0.9  # Should be close to both
    assert dot2 > 0.9
