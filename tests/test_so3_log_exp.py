import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PRECISIONS, identity_quaternion, random_quaternion
from scipy.spatial.transform import Rotation as R

from nanomanifold import SO3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_log_identity(backend, batch_dims, precision):
    # Log of identity should be zero vector
    identity = identity_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)
    log_result = SO3.log(identity)

    assert log_result.shape == batch_dims + (3,)

    if precision >= 32:
        log_np = np.array(log_result)
        assert np.allclose(log_np, 0.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_exp_zero(backend, batch_dims, precision):
    # Exp of zero vector should be identity quaternion
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend.replace("jax", "jax"))

    if precision == 16:
        dtype = xp.float16
    elif precision == 32:
        dtype = xp.float32
    else:
        dtype = xp.float64

    zero_vec = xp.zeros(batch_dims + (3,), dtype=dtype)
    exp_result = SO3.exp(zero_vec)

    identity = identity_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    assert exp_result.shape == identity.shape

    if precision >= 32:
        exp_np = np.array(exp_result)
        identity_np = np.array(identity)

        # Check quaternion equivalence (accounting for q/-q ambiguity)
        dot_products = np.sum(exp_np * identity_np, axis=-1)
        assert np.allclose(np.abs(dot_products), 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_log_exp_inverse(backend, batch_dims, precision):
    # Test that exp(log(q)) = q for random quaternions
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    log_quat = SO3.log(quat)
    exp_log_quat = SO3.exp(log_quat)

    assert exp_log_quat.shape == quat.shape

    if precision >= 32:
        quat_np = np.array(quat)
        result_np = np.array(exp_log_quat)

        # Check quaternion equivalence (q and -q represent same rotation)
        dot_products = np.sum(quat_np * result_np, axis=-1)
        assert np.allclose(np.abs(dot_products), 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_exp_log_inverse(backend, batch_dims, precision):
    # Test that log(exp(v)) = v for small tangent vectors
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend.replace("jax", "jax"))

    # Use small random tangent vectors (within the domain of the logarithm)
    # Generate in numpy first (following conftest pattern)
    shape = batch_dims + (3,)
    tangent_vec_np = 0.5 * np.random.normal(0, 1, size=shape).astype(f"float{precision}")
    tangent_vec = xp.asarray(tangent_vec_np)

    exp_vec = SO3.exp(tangent_vec)
    log_exp_vec = SO3.log(exp_vec)

    assert log_exp_vec.shape == tangent_vec.shape

    if precision >= 32:
        original_np = np.array(tangent_vec)
        result_np = np.array(log_exp_vec)
        assert np.allclose(original_np, result_np, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_log_exp_scipy(backend, batch_dims):
    # Compare with scipy for known cases
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=32)

    # Compute log using nanomanifold
    log_result = SO3.log(quat)

    # Convert to numpy for scipy
    quat_np = np.array(quat)

    # Convert from [w, x, y, z] to scipy's [x, y, z, w] format
    quat_scipy = np.concatenate([quat_np[..., 1:4], quat_np[..., 0:1]], axis=-1)

    # Reshape for scipy processing
    quat_flat = quat_scipy.reshape(-1, 4)

    # Compute axis-angle using scipy
    scipy_log = []
    for i in range(len(quat_flat)):
        r = R.from_quat(quat_flat[i : i + 1])
        # Get rotation vector (axis * angle)
        rotvec = r.as_rotvec()
        scipy_log.append(rotvec[0])

    scipy_log = np.array(scipy_log).reshape(batch_dims + (3,))

    assert log_result.shape == scipy_log.shape

    log_np = np.array(log_result)
    assert np.allclose(log_np, scipy_log, atol=1e-6)


def test_log_specific_rotations():
    # Test log for known rotations
    identity = np.array([1.0, 0.0, 0.0, 0.0])

    # 90-degree rotation around z-axis
    sqrt2_inv = 1.0 / np.sqrt(2.0)
    q_z90 = np.array([sqrt2_inv, 0, 0, sqrt2_inv])

    # 180-degree rotation around x-axis
    q_x180 = np.array([0.0, 1.0, 0.0, 0.0])

    # Test identity
    log_identity = SO3.log(identity)
    log_identity_np = np.array(log_identity)
    assert np.allclose(log_identity_np, [0.0, 0.0, 0.0], atol=1e-6)

    # Test 90-degree rotation around z
    log_z90 = SO3.log(q_z90)
    log_z90_np = np.array(log_z90)
    expected_z90 = [0.0, 0.0, np.pi / 2]
    assert np.allclose(log_z90_np, expected_z90, atol=1e-6)

    # Test 180-degree rotation around x
    log_x180 = SO3.log(q_x180)
    log_x180_np = np.array(log_x180)
    expected_x180 = [np.pi, 0.0, 0.0]
    assert np.allclose(log_x180_np, expected_x180, atol=1e-6)


def test_exp_specific_vectors():
    # Test exp for known tangent vectors
    zero_vec = np.array([0.0, 0.0, 0.0])

    # π/2 rotation around z-axis
    vec_z90 = np.array([0.0, 0.0, np.pi / 2])

    # π rotation around x-axis
    vec_x180 = np.array([np.pi, 0.0, 0.0])

    # Test zero vector
    exp_zero = SO3.exp(zero_vec)
    exp_zero_np = np.array(exp_zero)
    expected_identity = [1.0, 0.0, 0.0, 0.0]
    # Check quaternion equivalence
    dot_product = np.sum(exp_zero_np * expected_identity)
    assert np.allclose(np.abs(dot_product), 1.0, atol=1e-6)

    # Test 90-degree rotation around z
    exp_z90 = SO3.exp(vec_z90)
    exp_z90_np = np.array(exp_z90)
    sqrt2_inv = 1.0 / np.sqrt(2.0)
    expected_z90 = [sqrt2_inv, 0.0, 0.0, sqrt2_inv]
    dot_product = np.sum(exp_z90_np * expected_z90)
    assert np.allclose(np.abs(dot_product), 1.0, atol=1e-6)

    # Test 180-degree rotation around x
    exp_x180 = SO3.exp(vec_x180)
    exp_x180_np = np.array(exp_x180)
    expected_x180 = [0.0, 1.0, 0.0, 0.0]
    dot_product = np.sum(exp_x180_np * expected_x180)
    assert np.allclose(np.abs(dot_product), 1.0, atol=1e-6)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_log_differentiability_torch(batch_dims):
    torch = pytest.importorskip("torch")
    # Use double precision for gradient checking
    dtype = torch.float64

    # Random quaternion input (avoiding singularities)
    quat = random_quaternion(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)

    # Check gradients of SO3.log
    def f_log(q):
        return SO3.log(q).sum()

    assert torch.autograd.gradcheck(f_log, (quat,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_exp_differentiability_torch(batch_dims):
    torch = pytest.importorskip("torch")
    # Use double precision for gradient checking
    dtype = torch.float64

    # Random tangent vector input (small to avoid singularities) - generate in numpy first
    shape = batch_dims + (3,)
    tangent_vec_np = 0.5 * np.random.normal(0, 1, size=shape).astype("float64")
    tangent_vec = torch.tensor(tangent_vec_np, dtype=dtype, requires_grad=True)

    # Check gradients of SO3.exp
    def f_exp(v):
        return SO3.exp(v).sum()

    assert torch.autograd.gradcheck(f_exp, (tangent_vec,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_log_exp_jittability_jax(batch_dims):
    """Test that log and exp functions are JIT-compatible with JAX"""
    jax = pytest.importorskip("jax")

    # Create test quaternion and tangent vector
    q = random_quaternion(batch_dims=batch_dims, backend="jax", precision=32)
    # Create a small random tangent vector - generate in numpy first
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name("jax")
    shape = batch_dims + (3,)
    tangent_np = 0.1 * np.random.normal(0, 1, size=shape).astype("float32")
    tangent = xp.asarray(tangent_np)

    # Define JIT-compiled functions
    @jax.jit
    def jit_log(q):
        return SO3.log(q)

    @jax.jit
    def jit_exp(v):
        return SO3.exp(v)

    # Test that JIT compilation works and compare with non-JIT
    log_result_jit = jit_log(q)
    exp_result_jit = jit_exp(tangent)

    log_result_non_jit = SO3.log(q)
    exp_result_non_jit = SO3.exp(tangent)

    # Verify results match between JIT and non-JIT
    assert jax.numpy.allclose(log_result_jit, log_result_non_jit, atol=1e-6)
    assert jax.numpy.allclose(exp_result_jit, exp_result_non_jit, atol=1e-6)

    # Verify results have correct shapes
    assert log_result_jit.shape == batch_dims + (3,)
    assert exp_result_jit.shape == batch_dims + (4,)

    # Verify exp result is unit quaternion
    exp_norm = jax.numpy.linalg.norm(exp_result_jit, axis=-1)
    assert jax.numpy.allclose(exp_norm, 1.0, atol=1e-6)


def test_log_exp_composition():
    # Test that log and exp compose properly with other operations

    # Test: log(q1 * q2) should relate to log(q1) and log(q2) in a specific way
    # This is more complex for SO(3) due to non-commutativity, but we can test
    # that the operations are at least consistent

    q1 = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0])  # 90° around x
    q2 = np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0])  # 90° around y

    # Multiply quaternions
    q_product = SO3.multiply(q1, q2)

    # Log of the product
    log_product = SO3.log(q_product)

    # Exp of the log should give back the product
    exp_log_product = SO3.exp(log_product)

    product_np = np.array(q_product)
    result_np = np.array(exp_log_product)

    # Check quaternion equivalence
    dot_product = np.sum(product_np * result_np)
    assert np.allclose(np.abs(dot_product), 1.0, atol=1e-6)


def test_log_exp_broadcasting():
    # Test broadcasting between different batch sizes
    quat_single = random_quaternion(batch_dims=(), backend="numpy", precision=32)
    quat_batch = random_quaternion(batch_dims=(3,), backend="numpy", precision=32)

    # Test log broadcasting
    log_single = SO3.log(quat_single)
    log_batch = SO3.log(quat_batch)

    assert log_single.shape == (3,)
    assert log_batch.shape == (3, 3)

    # Test exp broadcasting
    tangent_single = np.array([0.1, 0.2, 0.3])
    tangent_batch = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

    exp_single = SO3.exp(tangent_single)
    exp_batch = SO3.exp(tangent_batch)

    assert exp_single.shape == (4,)
    assert exp_batch.shape == (3, 4)


def test_log_exp_edge_cases():
    # Test edge cases and numerical stability

    # Very small rotations
    small_vec = np.array([1e-8, 1e-8, 1e-8])
    q_small = SO3.exp(small_vec)
    log_small = SO3.log(q_small)

    small_vec_np = np.array(small_vec)
    log_small_np = np.array(log_small)
    assert np.allclose(small_vec_np, log_small_np, atol=1e-12)

    # Near π rotations (close to singularity)
    large_vec = np.array([np.pi - 1e-6, 0, 0])
    q_large = SO3.exp(large_vec)
    log_large = SO3.log(q_large)

    large_vec_np = np.array(large_vec)
    log_large_np = np.array(log_large)
    assert np.allclose(large_vec_np, log_large_np, atol=1e-5)
