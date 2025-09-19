import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PRECISIONS, identity_se3, random_points, random_se3

from nanomanifold import SE3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_inverse_identity(backend, batch_dims, precision):
    """Test that inverse of identity is identity."""
    identity = identity_se3(batch_dims=batch_dims, backend=backend, precision=precision)
    identity_inv = SE3.inverse(identity)

    assert identity_inv.dtype == identity.dtype
    assert identity_inv.shape == identity.shape

    # Convert to numpy arrays and compare
    identity_np = np.array(identity)
    identity_inv_np = np.array(identity_inv)

    # Check quaternion equivalence (q and -q represent the same rotation)
    quat_identity = identity_np[..., :4]
    quat_identity_inv = identity_inv_np[..., :4]
    dot_products = np.sum(quat_identity * quat_identity_inv, axis=-1)
    assert np.allclose(np.abs(dot_products), 1.0, atol=ATOL[precision])

    # Check translation is zero
    trans_identity = identity_np[..., 4:7]
    trans_identity_inv = identity_inv_np[..., 4:7]
    assert np.allclose(trans_identity, trans_identity_inv, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_inverse_involution(backend, batch_dims, precision):
    """Test that inverse of inverse is the original transformation."""
    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)
    se3_inv = SE3.inverse(se3)
    se3_inv_inv = SE3.inverse(se3_inv)

    assert se3_inv_inv.dtype == se3.dtype
    assert se3_inv_inv.shape == se3.shape

    # Convert to numpy arrays and compare
    se3_np = np.array(se3)
    se3_inv_inv_np = np.array(se3_inv_inv)

    # Check quaternion equivalence (q and -q represent the same rotation)
    quat_orig = se3_np[..., :4]
    quat_inv_inv = se3_inv_inv_np[..., :4]
    dot_products = np.sum(quat_orig * quat_inv_inv, axis=-1)
    assert np.allclose(np.abs(dot_products), 1.0, atol=ATOL[precision])

    # Check translation equivalence
    trans_orig = se3_np[..., 4:7]
    trans_inv_inv = se3_inv_inv_np[..., 4:7]
    assert np.allclose(trans_orig, trans_inv_inv, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_inverse_multiply_identity(backend, batch_dims, precision):
    """Test that SE3 * SE3^(-1) = Identity."""
    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)
    se3_inv = SE3.inverse(se3)

    # Test se3 * se3_inv = identity
    result1 = SE3.multiply(se3, se3_inv)
    # Test se3_inv * se3 = identity
    result2 = SE3.multiply(se3_inv, se3)

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
def test_inverse_matrix_equivalence(backend, batch_dims, precision):
    """Test that SE3 inverse matches matrix inverse."""
    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)
    se3_inv = SE3.inverse(se3)

    # Convert to matrices
    matrix = SE3.to_matrix(se3)
    SE3.to_matrix(se3_inv)

    # Compute matrix inverse using numpy (with float16 workaround)
    matrix_np = np.array(matrix)
    if matrix_np.dtype == np.float16:
        # numpy.linalg.inv doesn't support float16, so convert to float32
        matrix_f32 = matrix_np.astype(np.float32)
        matrix_inv_f32 = np.linalg.inv(matrix_f32)
        matrix_inv_numpy = matrix_inv_f32.astype(np.float16)
    else:
        matrix_inv_numpy = np.linalg.inv(matrix_np)

    # Convert numpy inverse back to SE3
    se3_from_matrix_inv = SE3.from_matrix(matrix_inv_numpy)

    assert se3_inv.dtype == se3.dtype
    assert se3_inv.shape == se3.shape

    # Convert to numpy arrays and compare
    se3_inv_np = np.array(se3_inv)
    se3_from_matrix_inv_np = np.array(se3_from_matrix_inv)

    # Check quaternion equivalence (q and -q represent the same rotation)
    quat_inv = se3_inv_np[..., :4]
    quat_from_matrix = se3_from_matrix_inv_np[..., :4]
    dot_products = np.sum(quat_inv * quat_from_matrix, axis=-1)
    assert np.allclose(np.abs(dot_products), 1.0, atol=ATOL[precision])

    # Check translation equivalence
    trans_inv = se3_inv_np[..., 4:7]
    trans_from_matrix = se3_from_matrix_inv_np[..., 4:7]
    assert np.allclose(trans_inv, trans_from_matrix, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_inverse_scipy(backend, batch_dims):
    """Test inverse against scipy spatial transform."""
    pytest.importorskip("scipy.spatial")
    from scipy.spatial.transform import Rotation as R

    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=32)
    se3_inv = SE3.inverse(se3)

    # Convert to numpy for scipy
    se3_np = np.array(se3)
    se3_inv_np = np.array(se3_inv)

    # Extract quaternions and translations
    q_np = se3_np[..., :4]
    t_np = se3_np[..., 4:7]
    se3_inv_np[..., :4]
    se3_inv_np[..., 4:7]

    # Convert from [w, x, y, z] to scipy's [x, y, z, w] format
    q_scipy = np.concatenate([q_np[..., 1:4], q_np[..., 0:1]], axis=-1)

    # Reshape for scipy processing
    q_flat = q_scipy.reshape(-1, 4)
    t_flat = t_np.reshape(-1, 3)

    # Compute inverse using scipy
    q_inv_scipy = []
    t_inv_scipy = []
    for i in range(len(q_flat)):
        r = R.from_quat(q_flat[i : i + 1])
        r_inv = r.inv()

        # For SE3 inverse: t_inv = -R_inv * t = -R^T * t
        t_inv_expected = -r_inv.apply(t_flat[i : i + 1])

        q_inv_scipy.append(r_inv.as_quat())
        t_inv_scipy.append(t_inv_expected)

    q_inv_scipy = np.array(q_inv_scipy).reshape(q_scipy.shape)
    t_inv_scipy = np.array(t_inv_scipy).reshape(t_np.shape)

    # Convert back to [w, x, y, z] format
    q_inv_scipy = np.concatenate([q_inv_scipy[..., 3:4], q_inv_scipy[..., 0:3]], axis=-1)

    if backend != "numpy":
        se3_inv_np = np.array(se3_inv)

    # Check quaternion equivalence (q and -q represent the same rotation)
    quat_inv = se3_inv_np[..., :4]
    dot_products = np.sum(quat_inv * q_inv_scipy, axis=-1)
    assert np.allclose(np.abs(dot_products), 1.0, atol=1e-6)

    # Check translation equivalence
    trans_inv = se3_inv_np[..., 4:7]
    assert np.allclose(trans_inv, t_inv_scipy, atol=1e-6)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_inverse_differentiability_torch(batch_dims):
    """Test that inverse function is differentiable with PyTorch."""
    torch = pytest.importorskip("torch")
    # Use double precision for gradient checking as recommended by PyTorch
    dtype = torch.float64

    # Random SE3 input
    se3 = random_se3(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)

    # Check gradients of SE3.inverse
    def f(se3):
        return SE3.inverse(se3)

    assert torch.autograd.gradcheck(f, (se3,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_inverse_jittability_jax(batch_dims):
    """Test that inverse function is JIT-compatible with JAX."""
    jax = pytest.importorskip("jax")

    # Create test SE3 transformation
    se3 = random_se3(batch_dims=batch_dims, backend="jax", precision=32)

    # Define JIT-compiled function
    @jax.jit
    def jit_inverse(se3):
        return SE3.inverse(se3)

    # Test that JIT compilation works and compare with non-JIT
    result_jit = jit_inverse(se3)
    result_non_jit = SE3.inverse(se3)

    # Verify results match between JIT and non-JIT
    assert jax.numpy.allclose(result_jit, result_non_jit, atol=1e-6)

    # Verify result has correct shape
    assert result_jit.shape == batch_dims + (7,)

    # Verify quaternion is unit
    quat_norm = jax.numpy.linalg.norm(result_jit[..., :4], axis=-1)
    assert jax.numpy.allclose(quat_norm, 1.0, atol=1e-6)


def test_inverse_specific_transformations():
    """Test inverse of specific known transformations."""
    # Pure translation: [1, 0, 0, 0, 1, 2, 3]
    t_xyz = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    t_xyz_inv = SE3.inverse(t_xyz)

    # Expected inverse: [1, 0, 0, 0, -1, -2, -3]
    expected_inv = np.array([1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -3.0])

    t_xyz_inv_np = np.array(t_xyz_inv)
    assert np.allclose(t_xyz_inv_np, expected_inv, atol=1e-6)

    # 90-degree rotation around z-axis with translation: [cos(π/4), 0, 0, sin(π/4), 1, 0, 0]
    sqrt2_inv = 1.0 / np.sqrt(2.0)
    r_z90_t = np.array([sqrt2_inv, 0.0, 0.0, sqrt2_inv, 1.0, 0.0, 0.0])
    r_z90_t_inv = SE3.inverse(r_z90_t)

    # Test by verifying that transformation * inverse = identity through point transformation
    identity_check = SE3.multiply(r_z90_t, r_z90_t_inv)
    identity_expected = identity_se3(batch_dims=(), backend="numpy", precision=32)

    identity_check_np = np.array(identity_check)
    identity_expected_np = np.array(identity_expected)

    # Check quaternion equivalence (q and -q represent the same rotation)
    quat_check = identity_check_np[..., :4]
    quat_expected = identity_expected_np[..., :4]
    dot_product = np.sum(quat_check * quat_expected, axis=-1)
    assert np.allclose(np.abs(dot_product), 1.0, atol=1e-6)

    # Check translation is zero
    trans_check = identity_check_np[..., 4:7]
    assert np.allclose(trans_check, 0.0, atol=1e-6)


def test_inverse_point_transformation():
    """Test that SE3 inverse correctly inverts point transformations."""
    # Create random SE3 transformation
    se3 = random_se3(batch_dims=(), backend="numpy", precision=32)
    se3_inv = SE3.inverse(se3)

    # Create random points
    points = random_points(batch_dims=(), n_points=10, backend="numpy", precision=32)

    # Transform points using SE3 transformation
    points_transformed = transform_points_se3(se3, points)

    # Transform back using inverse
    points_recovered = transform_points_se3(se3_inv, points_transformed)

    # Results should match original points
    points_np = np.array(points)
    points_recovered_np = np.array(points_recovered)

    assert np.allclose(points_np, points_recovered_np, atol=1e-6)


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
