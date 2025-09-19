import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PRECISIONS, identity_se3, random_se3

from nanomanifold import SE3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_log_identity(backend, batch_dims, precision):
    """Test that log of identity SE3 transformation gives zero tangent vector."""
    identity = identity_se3(batch_dims=batch_dims, backend=backend, precision=precision)
    log_result = SE3.log(identity)

    assert log_result.shape == batch_dims + (6,)

    log_np = np.array(log_result)
    assert np.allclose(log_np, 0.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_exp_zero(backend, batch_dims, precision):
    """Test that exp of zero tangent vector gives identity SE3 transformation."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend.replace("jax", "jax"))

    if precision == 16:
        dtype = xp.float16
    elif precision == 32:
        dtype = xp.float32
    else:
        dtype = xp.float64

    zero_tangent = xp.zeros(batch_dims + (6,), dtype=dtype)
    exp_result = SE3.exp(zero_tangent)

    identity = identity_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    assert exp_result.shape == identity.shape

    exp_np = np.array(exp_result)
    identity_np = np.array(identity)

    # Check quaternion equivalence (accounting for q/-q ambiguity)
    q_exp = exp_np[..., :4]
    q_identity = identity_np[..., :4]
    dot_products = np.sum(q_exp * q_identity, axis=-1)
    assert np.allclose(np.abs(dot_products), 1.0, atol=ATOL[precision])

    t_exp = exp_np[..., 4:7]
    t_identity = identity_np[..., 4:7]
    assert np.allclose(t_exp, t_identity, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_log_exp_inverse(backend, batch_dims, precision):
    """Test that exp(log(se3)) = se3 for random SE3 transformations."""
    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    log_se3 = SE3.log(se3)
    exp_log_se3 = SE3.exp(log_se3)

    assert exp_log_se3.shape == se3.shape

    se3_np = np.array(se3)
    result_np = np.array(exp_log_se3)

    # Check quaternion equivalence (q and -q represent same rotation)
    q_orig = se3_np[..., :4]
    q_result = result_np[..., :4]
    dot_products = np.sum(q_orig * q_result, axis=-1)
    assert np.allclose(np.abs(dot_products), 1.0, atol=ATOL[precision])

    t_orig = se3_np[..., 4:7]
    t_result = result_np[..., 4:7]
    assert np.allclose(t_orig, t_result, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_exp_log_inverse(backend, batch_dims, precision):
    """Test that log(exp(tangent)) = tangent for small tangent vectors."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend.replace("jax", "jax"))

    shape = batch_dims + (6,)
    tangent_vec_np = 0.5 * np.random.normal(0, 1, size=shape).astype(f"float{precision}")
    tangent_vec = xp.asarray(tangent_vec_np)

    exp_vec = SE3.exp(tangent_vec)
    log_exp_vec = SE3.log(exp_vec)

    assert log_exp_vec.shape == tangent_vec.shape

    original_np = np.array(tangent_vec)
    result_np = np.array(log_exp_vec)
    assert np.allclose(original_np, result_np, atol=ATOL[precision])


def test_log_specific_transformations():
    """Test log for known SE3 transformations."""
    se3_translation = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])

    sqrt2_inv = 1.0 / np.sqrt(2.0)
    se3_rotation = np.array([sqrt2_inv, 0.0, 0.0, sqrt2_inv, 0.0, 0.0, 0.0])

    se3_combined = np.array([sqrt2_inv, 0.0, 0.0, sqrt2_inv, 1.0, 2.0, 3.0])

    log_translation = SE3.log(se3_translation)
    log_translation_np = np.array(log_translation)
    expected_translation = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]  # [ω=0, ρ=t]
    assert np.allclose(log_translation_np, expected_translation, atol=1e-6)

    log_rotation = SE3.log(se3_rotation)
    log_rotation_np = np.array(log_rotation)
    expected_rotation = [0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0]  # [ω=z*π/2, ρ=0]
    assert np.allclose(log_rotation_np, expected_rotation, atol=1e-6)

    log_combined = SE3.log(se3_combined)
    log_combined_np = np.array(log_combined)
    # For combined, the translation part is transformed by the V^(-1) matrix
    # We mainly test that it's reasonable and invertible
    assert log_combined_np.shape == (6,)
    assert np.allclose(log_combined_np[:3], [0.0, 0.0, np.pi / 2], atol=1e-6)  # Rotation part unchanged


def test_exp_specific_tangents():
    """Test exp for known tangent vectors."""
    zero_tangent = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    omega_z90 = np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0])

    rho_translation = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])

    exp_zero = SE3.exp(zero_tangent)
    exp_zero_np = np.array(exp_zero)
    expected_identity = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Check quaternion equivalence and exact translation
    dot_product = np.sum(exp_zero_np[:4] * expected_identity[:4])
    assert np.allclose(np.abs(dot_product), 1.0, atol=1e-6)
    assert np.allclose(exp_zero_np[4:], expected_identity[4:], atol=1e-6)

    exp_rotation = SE3.exp(omega_z90)
    exp_rotation_np = np.array(exp_rotation)
    sqrt2_inv = 1.0 / np.sqrt(2.0)
    expected_rotation = [sqrt2_inv, 0.0, 0.0, sqrt2_inv, 0.0, 0.0, 0.0]
    dot_product = np.sum(exp_rotation_np[:4] * expected_rotation[:4])
    assert np.allclose(np.abs(dot_product), 1.0, atol=1e-6)
    assert np.allclose(exp_rotation_np[4:], expected_rotation[4:], atol=1e-6)

    exp_translation = SE3.exp(rho_translation)
    exp_translation_np = np.array(exp_translation)
    expected_translation = [1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
    dot_product = np.sum(exp_translation_np[:4] * expected_translation[:4])
    assert np.allclose(np.abs(dot_product), 1.0, atol=1e-6)
    assert np.allclose(exp_translation_np[4:], expected_translation[4:], atol=1e-6)


def test_log_exp_matrix_consistency():
    """Test that log/exp is consistent with matrix conversion."""
    se3 = random_se3(batch_dims=(), backend="numpy", precision=32)

    matrix = SE3.to_matrix(se3)
    se3_from_matrix = SE3.from_matrix(matrix)

    log_orig = SE3.log(se3)
    log_from_matrix = SE3.log(se3_from_matrix)

    log_orig_np = np.array(log_orig)
    log_from_matrix_np = np.array(log_from_matrix)

    assert np.allclose(log_orig_np, log_from_matrix_np, atol=1e-6)


def test_log_exp_composition():
    """Test that log and exp compose properly with other operations."""
    se3_1 = random_se3(batch_dims=(), backend="numpy", precision=32)
    se3_2 = random_se3(batch_dims=(), backend="numpy", precision=32)

    se3_product = SE3.multiply(se3_1, se3_2)

    log_product = SE3.log(se3_product)

    exp_log_product = SE3.exp(log_product)

    product_np = np.array(se3_product)
    result_np = np.array(exp_log_product)

    # Check quaternion equivalence
    q_prod = product_np[:4]
    q_result = result_np[:4]
    dot_product = np.sum(q_prod * q_result)
    assert np.allclose(np.abs(dot_product), 1.0, atol=1e-6)

    # Check translation equivalence
    t_prod = product_np[4:]
    t_result = result_np[4:]
    assert np.allclose(t_prod, t_result, atol=1e-6)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_log_differentiability_torch(batch_dims):
    """Test that log function is differentiable with PyTorch."""
    torch = pytest.importorskip("torch")
    # Use double precision for gradient checking
    dtype = torch.float64

    # Random SE3 input (avoiding singularities)
    se3 = random_se3(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)

    def f_log(se3):
        return SE3.log(se3).sum()

    assert torch.autograd.gradcheck(f_log, (se3,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_exp_differentiability_torch(batch_dims):
    """Test that exp function is differentiable with PyTorch."""
    torch = pytest.importorskip("torch")
    dtype = torch.float64

    shape = batch_dims + (6,)
    tangent_vec_np = 0.5 * np.random.normal(0, 1, size=shape).astype("float64")
    tangent_vec = torch.tensor(tangent_vec_np, dtype=dtype, requires_grad=True)

    def f_exp(v):
        return SE3.exp(v).sum()

    assert torch.autograd.gradcheck(f_exp, (tangent_vec,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_log_exp_jittability_jax(batch_dims):
    """Test that log and exp functions are JIT-compatible with JAX."""
    jax = pytest.importorskip("jax")
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name("jax")

    se3 = random_se3(batch_dims=batch_dims, backend="jax", precision=32)
    shape = batch_dims + (6,)
    tangent_np = 0.1 * np.random.normal(0, 1, size=shape).astype("float32")
    tangent = xp.asarray(tangent_np)

    @jax.jit
    def jit_log(se3):
        return SE3.log(se3)

    @jax.jit
    def jit_exp(v):
        return SE3.exp(v)

    log_result_jit = jit_log(se3)
    exp_result_jit = jit_exp(tangent)

    log_result_non_jit = SE3.log(se3)
    exp_result_non_jit = SE3.exp(tangent)

    assert jax.numpy.allclose(log_result_jit, log_result_non_jit, atol=1e-6)
    assert jax.numpy.allclose(exp_result_jit, exp_result_non_jit, atol=1e-6)

    assert log_result_jit.shape == batch_dims + (6,)
    assert exp_result_jit.shape == batch_dims + (7,)

    q_result = exp_result_jit[..., :4]
    q_norm = jax.numpy.linalg.norm(q_result, axis=-1)
    assert jax.numpy.allclose(q_norm, 1.0, atol=1e-6)


def test_log_exp_broadcasting():
    """Test broadcasting between different batch sizes."""
    se3_single = random_se3(batch_dims=(), backend="numpy", precision=32)

    tangent_batch = 0.1 * np.random.normal(0, 1, size=(3, 6)).astype("float32")

    log_single = SE3.log(se3_single)
    assert log_single.shape == (6,)

    exp_batch = SE3.exp(tangent_batch)
    assert exp_batch.shape == (3, 7)

    exp_batch_np = np.array(exp_batch)
    assert np.all(np.isfinite(exp_batch_np))

    q_norms = np.linalg.norm(exp_batch_np[:, :4], axis=-1)
    assert np.allclose(q_norms, 1.0, atol=1e-6)


def test_log_exp_edge_cases():
    """Test edge cases and numerical stability."""
    small_tangent = np.array([1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8])
    se3_small = SE3.exp(small_tangent)
    log_small = SE3.log(se3_small)

    small_tangent_np = np.array(small_tangent)
    log_small_np = np.array(log_small)
    assert np.allclose(small_tangent_np, log_small_np, atol=1e-12)

    large_tangent = np.array([np.pi - 1e-6, 0, 0, 1.0, 2.0, 3.0])
    se3_large = SE3.exp(large_tangent)
    log_large = SE3.log(se3_large)

    large_tangent_np = np.array(large_tangent)
    log_large_np = np.array(log_large)
    assert np.allclose(large_tangent_np, log_large_np, atol=1e-5)

    pure_rotation_tangent = np.array([0.5, 0.3, 0.1, 0.0, 0.0, 0.0])
    se3_pure_rot = SE3.exp(pure_rotation_tangent)
    log_pure_rot = SE3.log(se3_pure_rot)

    pure_rotation_np = np.array(pure_rotation_tangent)
    log_pure_rot_np = np.array(log_pure_rot)
    assert np.allclose(pure_rotation_np, log_pure_rot_np, atol=1e-6)

    pure_translation_tangent = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    se3_pure_trans = SE3.exp(pure_translation_tangent)
    log_pure_trans = SE3.log(se3_pure_trans)

    pure_translation_np = np.array(pure_translation_tangent)
    log_pure_trans_np = np.array(log_pure_trans)
    assert np.allclose(pure_translation_np, log_pure_trans_np, atol=1e-6)
