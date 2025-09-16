import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PRECISIONS, identity_quaternion, random_quaternion
from scipy.spatial.transform import Rotation as R

from nanomanifold import SO3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_multiply_identity(backend, batch_dims, precision):
    # Create random quaternion and identity
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)
    identity = identity_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    # Test identity * quat = quat
    result1 = SO3.multiply(identity, quat)
    # Test quat * identity = quat
    result2 = SO3.multiply(quat, identity)

    assert result1.dtype == quat.dtype
    assert result1.shape == quat.shape
    assert result2.dtype == quat.dtype
    assert result2.shape == quat.shape

    # Convert to numpy arrays and compare
    quat_np = np.array(quat)
    result1_np = np.array(result1)
    result2_np = np.array(result2)

    if precision >= 32:
        # Check quaternion equivalence (q and -q represent the same rotation)
        dot_products1 = np.sum(quat_np * result1_np, axis=-1)
        dot_products2 = np.sum(quat_np * result2_np, axis=-1)
        assert np.allclose(np.abs(dot_products1), 1.0, atol=ATOL[precision])
        assert np.allclose(np.abs(dot_products2), 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_multiply_inverse(backend, batch_dims, precision):
    # Create random quaternion
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)
    quat_inv = SO3.inverse(quat)

    # Test quat * quat_inv = identity
    result1 = SO3.multiply(quat, quat_inv)
    # Test quat_inv * quat = identity
    result2 = SO3.multiply(quat_inv, quat)

    assert result1.dtype == quat.dtype
    assert result1.shape == quat.shape
    assert result2.dtype == quat.dtype
    assert result2.shape == quat.shape

    # Convert to numpy arrays and compare with identity quaternion
    identity = identity_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)
    identity_np = np.array(identity)
    result1_np = np.array(result1)
    result2_np = np.array(result2)

    if precision >= 32:
        # Check quaternion equivalence (q and -q represent the same rotation)
        dot_products1 = np.sum(identity_np * result1_np, axis=-1)
        dot_products2 = np.sum(identity_np * result2_np, axis=-1)
        assert np.allclose(np.abs(dot_products1), 1.0, atol=ATOL[precision])
        assert np.allclose(np.abs(dot_products2), 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_multiply_matrix_equivalence(backend, batch_dims, precision):
    # Create two random quaternions
    q1 = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)
    q2 = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    # Multiply quaternions
    q_result = SO3.multiply(q1, q2)

    # Convert to matrices and multiply
    R1 = SO3.to_matrix(q1)
    R2 = SO3.to_matrix(q2)
    R_result = np.matmul(np.array(R1), np.array(R2))

    # Convert result matrix back to quaternion
    q_from_matrix = SO3.from_matrix(R_result)

    assert q_result.dtype == q1.dtype
    assert q_result.shape == q1.shape

    # Convert to numpy arrays and compare
    q_result_np = np.array(q_result)
    q_from_matrix_np = np.array(q_from_matrix)

    if precision >= 32:
        # Check quaternion equivalence (q and -q represent the same rotation)
        dot_products = np.sum(q_result_np * q_from_matrix_np, axis=-1)
        assert np.allclose(np.abs(dot_products), 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_multiply_scipy(backend, batch_dims):
    # Create two random quaternions
    q1 = random_quaternion(batch_dims=batch_dims, backend=backend, precision=32)
    q2 = random_quaternion(batch_dims=batch_dims, backend=backend, precision=32)

    # Multiply using nanomanifold
    q_result = SO3.multiply(q1, q2)

    # Convert to numpy for scipy
    q1_np = np.array(q1)
    q2_np = np.array(q2)

    # Convert from [w, x, y, z] to scipy's [x, y, z, w] format
    q1_scipy = np.concatenate([q1_np[..., 1:4], q1_np[..., 0:1]], axis=-1)
    q2_scipy = np.concatenate([q2_np[..., 1:4], q2_np[..., 0:1]], axis=-1)

    # Reshape for scipy processing
    q1_flat = q1_scipy.reshape(-1, 4)
    q2_flat = q2_scipy.reshape(-1, 4)

    # Multiply using scipy
    q_result_scipy = []
    for i in range(len(q1_flat)):
        r1 = R.from_quat(q1_flat[i : i + 1])
        r2 = R.from_quat(q2_flat[i : i + 1])
        # For scipy, r1 * r2 means apply r2 first, then r1 (same as our convention)
        r_result = r1 * r2
        q_result_scipy.append(r_result.as_quat())

    q_result_scipy = np.array(q_result_scipy).reshape(q1_scipy.shape)
    # Convert back to [w, x, y, z] format
    q_result_scipy = np.concatenate([q_result_scipy[..., 3:4], q_result_scipy[..., 0:3]], axis=-1)

    assert q_result.dtype == q1.dtype
    assert q_result.shape == q_result_scipy.shape

    q_result_np = np.array(q_result)
    # Check quaternion equivalence (q and -q represent the same rotation)
    dot_products = np.sum(q_result_np * q_result_scipy, axis=-1)
    assert np.allclose(np.abs(dot_products), 1.0, atol=1e-6)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_multiply_differentiability_torch(batch_dims):
    torch = pytest.importorskip("torch")
    # Use double precision for gradient checking as recommended by PyTorch
    dtype = torch.float64

    # Random quaternion inputs
    q1 = random_quaternion(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)
    q2 = random_quaternion(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)

    # Check gradients of SO3.multiply with respect to first argument
    def f_q1(q):
        return SO3.multiply(q, q2.detach())

    assert torch.autograd.gradcheck(f_q1, (q1,), eps=1e-6, atol=1e-5)

    # Check gradients of SO3.multiply with respect to second argument
    def f_q2(q):
        return SO3.multiply(q1.detach(), q)

    assert torch.autograd.gradcheck(f_q2, (q2,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_multiply_jittability_jax(batch_dims):
    """Test that multiply function is JIT-compatible with JAX"""
    jax = pytest.importorskip("jax")

    # Create test quaternions
    q1 = random_quaternion(batch_dims=batch_dims, backend="jax", precision=32)
    q2 = random_quaternion(batch_dims=batch_dims, backend="jax", precision=32)

    # Define JIT-compiled function
    @jax.jit
    def jit_multiply(q1, q2):
        return SO3.multiply(q1, q2)

    # Test that JIT compilation works and compare with non-JIT
    result_jit = jit_multiply(q1, q2)
    result_non_jit = SO3.multiply(q1, q2)

    # Verify results match between JIT and non-JIT
    assert jax.numpy.allclose(result_jit, result_non_jit, atol=1e-6)

    # Verify result has correct shape and is unit quaternion
    assert result_jit.shape == batch_dims + (4,)
    result_norm = jax.numpy.linalg.norm(result_jit, axis=-1)
    assert jax.numpy.allclose(result_norm, 1.0, atol=1e-6)


def test_multiply_associativity():
    """Test that quaternion multiplication is associative: (q1 * q2) * q3 = q1 * (q2 * q3)"""
    # Create three random quaternions
    q1 = random_quaternion(batch_dims=(), backend="numpy", precision=32)
    q2 = random_quaternion(batch_dims=(), backend="numpy", precision=32)
    q3 = random_quaternion(batch_dims=(), backend="numpy", precision=32)

    # Compute (q1 * q2) * q3
    temp1 = SO3.multiply(q1, q2)
    result1 = SO3.multiply(temp1, q3)

    # Compute q1 * (q2 * q3)
    temp2 = SO3.multiply(q2, q3)
    result2 = SO3.multiply(q1, temp2)

    # Convert to numpy arrays and compare
    result1_np = np.array(result1)
    result2_np = np.array(result2)

    # Check quaternion equivalence (q and -q represent the same rotation)
    dot_product = np.sum(result1_np * result2_np, axis=-1)
    assert np.allclose(np.abs(dot_product), 1.0, atol=1e-6)


def test_multiply_specific_rotations():
    """Test multiplication of specific known rotations."""
    # 90-degree rotation around z-axis: [cos(π/4), 0, 0, sin(π/4)]
    sqrt2_inv = 1.0 / np.sqrt(2.0)
    q_z90 = np.array([sqrt2_inv, 0, 0, sqrt2_inv])

    # 90-degree rotation around x-axis: [cos(π/4), sin(π/4), 0, 0]
    q_x90 = np.array([sqrt2_inv, sqrt2_inv, 0, 0])

    # Multiply: first rotate around z, then around x
    q_result = SO3.multiply(q_x90, q_z90)

    # Verify by checking what happens to the point (1, 0, 0)
    point = np.array([[1.0, 0.0, 0.0]])

    # Expected: (1,0,0) -> z90 -> (0,1,0) -> x90 -> (0,0,1)
    rotated = SO3.rotate_points(q_result, point)
    rotated_np = np.array(rotated)
    expected = np.array([[0.0, 0.0, 1.0]])

    assert np.allclose(rotated_np, expected, atol=1e-6)


def test_multiply_broadcasting():
    """Test broadcasting between different batch sizes."""
    # Single quaternion and batch of quaternions
    q_single = random_quaternion(batch_dims=(), backend="numpy", precision=32)
    q_batch = random_quaternion(batch_dims=(3,), backend="numpy", precision=32)

    # Test broadcasting
    result1 = SO3.multiply(q_single, q_batch)
    result2 = SO3.multiply(q_batch, q_single)

    assert result1.shape == (3, 4)
    assert result2.shape == (3, 4)

    # Results should be different due to non-commutative multiplication
    result1_np = np.array(result1)
    result2_np = np.array(result2)

    # They shouldn't be equal (quaternion multiplication is not commutative)
    # Just check they have the right shapes and are valid quaternions
    norms1 = np.linalg.norm(result1_np, axis=-1)
    norms2 = np.linalg.norm(result2_np, axis=-1)
    assert np.allclose(norms1, 1.0, atol=1e-6)
    assert np.allclose(norms2, 1.0, atol=1e-6)
