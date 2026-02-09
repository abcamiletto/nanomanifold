import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PASS_XP, TEST_PRECISIONS, get_xp_kwargs, identity_se3, random_points, random_se3

from nanomanifold import SE3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_multiply_identity(backend, batch_dims, precision, pass_xp):
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    # Create random SE3 transformation and identity
    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)
    identity = identity_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    # Test identity * se3 = se3
    result1 = SE3.multiply(identity, se3, **xp_kwargs)
    # Test se3 * identity = se3
    result2 = SE3.multiply(se3, identity, **xp_kwargs)

    assert result1.dtype == se3.dtype
    assert result1.shape == se3.shape
    assert result2.dtype == se3.dtype
    assert result2.shape == se3.shape

    # Convert to numpy arrays and compare
    se3_np = np.array(se3)
    result1_np = np.array(result1)
    result2_np = np.array(result2)

    # Check quaternion equivalence (q and -q represent the same rotation)
    quat_orig = se3_np[..., :4]
    quat_result1 = result1_np[..., :4]
    quat_result2 = result2_np[..., :4]

    dot_products1 = np.sum(quat_orig * quat_result1, axis=-1)
    dot_products2 = np.sum(quat_orig * quat_result2, axis=-1)
    assert np.allclose(np.abs(dot_products1), 1.0, atol=ATOL[precision])
    assert np.allclose(np.abs(dot_products2), 1.0, atol=ATOL[precision])

    # Check translation equivalence
    trans_orig = se3_np[..., 4:7]
    trans_result1 = result1_np[..., 4:7]
    trans_result2 = result2_np[..., 4:7]
    assert np.allclose(trans_orig, trans_result1, atol=ATOL[precision])
    assert np.allclose(trans_orig, trans_result2, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_multiply_inverse(backend, batch_dims, precision, pass_xp):
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    # Create random SE3 transformation
    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    # Compute inverse using SE3.inverse
    se3_inv = SE3.inverse(se3, **xp_kwargs)

    # Test se3 * se3_inv = identity
    result1 = SE3.multiply(se3, se3_inv, **xp_kwargs)
    # Test se3_inv * se3 = identity
    result2 = SE3.multiply(se3_inv, se3, **xp_kwargs)

    assert result1.dtype == se3.dtype
    assert result1.shape == se3.shape
    assert result2.dtype == se3.dtype
    assert result2.shape == se3.shape

    # Convert to numpy arrays and compare with identity SE3
    identity = identity_se3(batch_dims=batch_dims, backend=backend, precision=precision)
    identity_np = np.array(identity)
    result1_np = np.array(result1)
    result2_np = np.array(result2)

    # Check quaternion equivalence (q and -q represent the same rotation)
    quat_identity = identity_np[..., :4]
    quat_result1 = result1_np[..., :4]
    quat_result2 = result2_np[..., :4]

    dot_products1 = np.sum(quat_identity * quat_result1, axis=-1)
    dot_products2 = np.sum(quat_identity * quat_result2, axis=-1)
    assert np.allclose(np.abs(dot_products1), 1.0, atol=ATOL[precision])
    assert np.allclose(np.abs(dot_products2), 1.0, atol=ATOL[precision])

    # Check translation is zero
    trans_result1 = result1_np[..., 4:7]
    trans_result2 = result2_np[..., 4:7]
    assert np.allclose(trans_result1, 0.0, atol=ATOL[precision])
    assert np.allclose(trans_result2, 0.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_multiply_matrix_equivalence(backend, batch_dims, precision, pass_xp):
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    # Create two random SE3 transformations
    se3_1 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)
    se3_2 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    # Multiply SE3 transformations
    se3_result = SE3.multiply(se3_1, se3_2, **xp_kwargs)

    # Convert to matrices and multiply
    matrix1 = SE3.to_matrix(se3_1, **xp_kwargs)
    matrix2 = SE3.to_matrix(se3_2, **xp_kwargs)
    matrix_result = np.matmul(np.array(matrix1), np.array(matrix2))

    # Convert result matrix back to SE3
    se3_from_matrix = SE3.from_matrix(matrix_result)

    assert se3_result.dtype == se3_1.dtype
    assert se3_result.shape == se3_1.shape

    # Convert to numpy arrays and compare
    se3_result_np = np.array(se3_result)
    se3_from_matrix_np = np.array(se3_from_matrix)

    # Check quaternion equivalence (q and -q represent the same rotation)
    quat_result = se3_result_np[..., :4]
    quat_from_matrix = se3_from_matrix_np[..., :4]
    dot_products = np.sum(quat_result * quat_from_matrix, axis=-1)
    assert np.allclose(np.abs(dot_products), 1.0, atol=ATOL[precision])

    # Check translation equivalence
    trans_result = se3_result_np[..., 4:7]
    trans_from_matrix = se3_from_matrix_np[..., 4:7]
    assert np.allclose(trans_result, trans_from_matrix, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_multiply_scipy(backend, batch_dims, pass_xp):
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    pytest.importorskip("scipy.spatial")
    from scipy.spatial.transform import Rotation as R

    # Create two random SE3 transformations
    se3_1 = random_se3(batch_dims=batch_dims, backend=backend, precision=32)
    se3_2 = random_se3(batch_dims=batch_dims, backend=backend, precision=32)

    # Multiply using nanomanifold
    se3_result = SE3.multiply(se3_1, se3_2, **xp_kwargs)

    # Convert to numpy for scipy
    se3_1_np = np.array(se3_1)
    se3_2_np = np.array(se3_2)

    # Extract quaternions and translations
    q1_np = se3_1_np[..., :4]
    t1_np = se3_1_np[..., 4:7]
    q2_np = se3_2_np[..., :4]
    t2_np = se3_2_np[..., 4:7]

    # Convert from [w, x, y, z] to scipy's [x, y, z, w] format
    q1_scipy = np.concatenate([q1_np[..., 1:4], q1_np[..., 0:1]], axis=-1)
    q2_scipy = np.concatenate([q2_np[..., 1:4], q2_np[..., 0:1]], axis=-1)

    # Reshape for scipy processing
    q1_flat = q1_scipy.reshape(-1, 4)
    q2_flat = q2_scipy.reshape(-1, 4)
    t1_flat = t1_np.reshape(-1, 3)
    t2_flat = t2_np.reshape(-1, 3)

    # Multiply using scipy (SE3 multiplication: result = se3_1 * se3_2)
    q_result_scipy = []
    t_result_scipy = []
    for i in range(len(q1_flat)):
        r1 = R.from_quat(q1_flat[i : i + 1])
        r2 = R.from_quat(q2_flat[i : i + 1])
        # For SE3: T_result = T1 * T2, where T = [R t; 0 1]
        # R_result = R1 * R2, t_result = R1 * t2 + t1
        r_result = r1 * r2
        t_result = r1.apply(t2_flat[i : i + 1]) + t1_flat[i : i + 1]

        q_result_scipy.append(r_result.as_quat())
        t_result_scipy.append(t_result)

    q_result_scipy = np.array(q_result_scipy).reshape(q1_scipy.shape)
    t_result_scipy = np.array(t_result_scipy).reshape(t1_np.shape)

    # Convert back to [w, x, y, z] format
    q_result_scipy = np.concatenate([q_result_scipy[..., 3:4], q_result_scipy[..., 0:3]], axis=-1)

    assert se3_result.dtype == se3_1.dtype
    se3_result_np = np.array(se3_result)

    # Check quaternion equivalence (q and -q represent the same rotation)
    quat_result = se3_result_np[..., :4]
    dot_products = np.sum(quat_result * q_result_scipy, axis=-1)
    assert np.allclose(np.abs(dot_products), 1.0, atol=1e-6)

    # Check translation equivalence
    trans_result = se3_result_np[..., 4:7]
    assert np.allclose(trans_result, t_result_scipy, atol=1e-6)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_multiply_differentiability_torch(batch_dims):
    torch = pytest.importorskip("torch")
    # Use double precision for gradient checking as recommended by PyTorch
    dtype = torch.float64

    # Random SE3 inputs
    se3_1 = random_se3(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)
    se3_2 = random_se3(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)

    # Check gradients of SE3.multiply with respect to first argument
    def f_se3_1(se3):
        return SE3.multiply(se3, se3_2.detach())

    assert torch.autograd.gradcheck(f_se3_1, (se3_1,), eps=1e-6, atol=1e-5)

    # Check gradients of SE3.multiply with respect to second argument
    def f_se3_2(se3):
        return SE3.multiply(se3_1.detach(), se3)

    assert torch.autograd.gradcheck(f_se3_2, (se3_2,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_multiply_jittability_jax(batch_dims):
    """Test that multiply function is JIT-compatible with JAX"""
    jax = pytest.importorskip("jax")

    # Create test SE3 transformations
    se3_1 = random_se3(batch_dims=batch_dims, backend="jax", precision=32)
    se3_2 = random_se3(batch_dims=batch_dims, backend="jax", precision=32)

    # Define JIT-compiled function
    @jax.jit
    def jit_multiply(se3_1, se3_2):
        return SE3.multiply(se3_1, se3_2)

    # Test that JIT compilation works and compare with non-JIT
    result_jit = jit_multiply(se3_1, se3_2)
    result_non_jit = SE3.multiply(se3_1, se3_2)

    # Verify results match between JIT and non-JIT
    assert jax.numpy.allclose(result_jit, result_non_jit, atol=1e-6)

    # Verify result has correct shape
    assert result_jit.shape == batch_dims + (7,)

    # Verify quaternion is unit
    quat_norm = jax.numpy.linalg.norm(result_jit[..., :4], axis=-1)
    assert jax.numpy.allclose(quat_norm, 1.0, atol=1e-6)


def test_multiply_associativity():
    """Test that SE3 multiplication is associative: (T1 * T2) * T3 = T1 * (T2 * T3)"""
    # Create three random SE3 transformations
    se3_1 = random_se3(batch_dims=(), backend="numpy", precision=32)
    se3_2 = random_se3(batch_dims=(), backend="numpy", precision=32)
    se3_3 = random_se3(batch_dims=(), backend="numpy", precision=32)

    # Compute (se3_1 * se3_2) * se3_3
    temp1 = SE3.multiply(se3_1, se3_2)
    result1 = SE3.multiply(temp1, se3_3)

    # Compute se3_1 * (se3_2 * se3_3)
    temp2 = SE3.multiply(se3_2, se3_3)
    result2 = SE3.multiply(se3_1, temp2)

    # Convert to numpy arrays and compare
    result1_np = np.array(result1)
    result2_np = np.array(result2)

    # Check quaternion equivalence (q and -q represent the same rotation)
    quat_result1 = result1_np[..., :4]
    quat_result2 = result2_np[..., :4]
    dot_product = np.sum(quat_result1 * quat_result2, axis=-1)
    assert np.allclose(np.abs(dot_product), 1.0, atol=1e-6)

    # Check translation equivalence
    trans_result1 = result1_np[..., 4:7]
    trans_result2 = result2_np[..., 4:7]
    assert np.allclose(trans_result1, trans_result2, atol=1e-6)


def test_multiply_specific_transformations():
    """Test multiplication of specific known transformations."""
    # Translation by (1, 0, 0): [1, 0, 0, 0, 1, 0, 0]
    t_x = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    # Translation by (0, 1, 0): [1, 0, 0, 0, 0, 1, 0]
    np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    # 90-degree rotation around z-axis: [cos(π/4), 0, 0, sin(π/4), 0, 0, 0]
    sqrt2_inv = 1.0 / np.sqrt(2.0)
    r_z90 = np.array([sqrt2_inv, 0.0, 0.0, sqrt2_inv, 0.0, 0.0, 0.0])

    # Test composition: first translate in x, then rotate around z
    # Expected: point at (2, 0, 0) should become (0, 2, 0)
    result = SE3.multiply(r_z90, t_x)

    # Test by transforming a point
    point = np.array([[2.0, 0.0, 0.0]])

    # Apply transformation sequence manually
    # First translate: (2,0,0) + (1,0,0) = (3,0,0)
    # Then rotate 90° around z: (3,0,0) -> (0,3,0)
    expected = np.array([[0.0, 3.0, 0.0]])

    # Apply using SE3 transformation matrix
    matrix = SE3.to_matrix(result)
    point_homogeneous = np.concatenate([point, np.ones((1, 1))], axis=-1)
    transformed_homogeneous = np.matmul(np.array(matrix), point_homogeneous.T).T
    transformed = transformed_homogeneous[:, :3]

    assert np.allclose(transformed, expected, atol=1e-6)


def test_multiply_point_transformation():
    """Test that SE3 multiplication correctly composes point transformations."""
    # Create two random SE3 transformations
    se3_1 = random_se3(batch_dims=(), backend="numpy", precision=32)
    se3_2 = random_se3(batch_dims=(), backend="numpy", precision=32)

    # Multiply them
    se3_composed = SE3.multiply(se3_1, se3_2)

    # Create random points
    points = random_points(batch_dims=(), n_points=10, backend="numpy", precision=32)

    # Transform points using composed transformation
    points_composed = transform_points_se3(se3_composed, points)

    # Transform points using individual transformations
    points_step1 = transform_points_se3(se3_2, points)
    points_step2 = transform_points_se3(se3_1, points_step1)

    # Results should match
    points_composed_np = np.array(points_composed)
    points_step2_np = np.array(points_step2)

    assert np.allclose(points_composed_np, points_step2_np, atol=1e-6)


def transform_points_se3(se3, points):
    """Helper function to transform points using SE3 transformation matrix."""
    matrix = SE3.to_matrix(se3)
    matrix_np = np.array(matrix)
    points_np = np.array(points)

    # Add homogeneous coordinate
    ones = np.ones(points_np.shape[:-1] + (1,))
    points_homogeneous = np.concatenate([points_np, ones], axis=-1)

    # Transform points
    if matrix_np.ndim == 2:  # Single transformation
        transformed_homogeneous = np.matmul(matrix_np, points_homogeneous.T).T
    else:  # Batched transformations
        transformed_homogeneous = np.matmul(matrix_np, points_homogeneous[..., :, None]).squeeze(-1)

    # Remove homogeneous coordinate
    return transformed_homogeneous[..., :3]
