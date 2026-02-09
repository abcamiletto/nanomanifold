import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PASS_XP, TEST_PRECISIONS, get_xp_kwargs, identity_se3, random_points, random_se3

from nanomanifold import SE3, SO3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_transform_points_identity(backend, batch_dims, precision, pass_xp):
    """Test that identity SE3 transformation doesn't change points."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    # Create identity SE3 transformation [1, 0, 0, 0, 0, 0, 0]
    identity = identity_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    # Generate random points
    points = random_points(batch_dims=batch_dims, n_points=10, backend=backend, precision=precision)

    # Apply transformation
    transformed_points = SE3.transform_points(identity, points, **xp_kwargs)

    assert transformed_points.dtype == points.dtype
    assert transformed_points.shape == points.shape

    # Convert to numpy arrays and compare
    points_np = np.array(points)
    transformed_points_np = np.array(transformed_points)

    assert np.allclose(points_np, transformed_points_np, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_transform_points_inverse(backend, batch_dims, precision, pass_xp):
    """Test that transforming by SE3 then by SE3^(-1) gives identity."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    # Create random SE3 transformation
    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    # Create inverse SE3 transformation using SE3.inverse()
    se3_inv = SE3.inverse(se3, **xp_kwargs)

    # Generate random points
    points = random_points(batch_dims=batch_dims, n_points=10, backend=backend, precision=precision)

    # Transform by se3 then by se3^(-1)
    transformed_points = SE3.transform_points(se3, points, **xp_kwargs)
    restored_points = SE3.transform_points(se3_inv, transformed_points, **xp_kwargs)

    assert restored_points.dtype == points.dtype
    assert restored_points.shape == points.shape

    # Convert to numpy arrays and compare
    points_np = np.array(points)
    restored_points_np = np.array(restored_points)

    # Use more lenient tolerance for complex batch dimensions due to accumulated floating-point error
    assert np.allclose(points_np, restored_points_np, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_transform_points_matrix_equivalence(backend, batch_dims, precision, pass_xp):
    """Test that SE3 point transformation matches matrix transformation."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    # Create random SE3 transformation
    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    # Generate random points
    points = random_points(batch_dims=batch_dims, n_points=10, backend=backend, precision=precision)

    # Transform points using SE3.transform_points
    transformed_points = SE3.transform_points(se3, points, **xp_kwargs)

    # Transform points using matrix approach
    matrix = SE3.to_matrix(se3, **xp_kwargs)
    matrix_np = np.array(matrix)
    points_np = np.array(points)

    # Add homogeneous coordinate (w=1)
    ones = np.ones(points_np.shape[:-1] + (1,))
    points_homogeneous = np.concatenate([points_np, ones], axis=-1)  # (..., N, 4)

    # Apply matrix transformation
    if matrix_np.ndim == 2:  # Single transformation
        # For single transformation: (N,4) @ (4,4)^T -> (N,4)
        transformed_homogeneous = np.matmul(points_homogeneous, matrix_np.T)
    else:  # Batched transformations
        # Einstein summation for batched matrix multiplication
        transformed_homogeneous = np.einsum("...ij,...nj->...ni", matrix_np, points_homogeneous)

    # Remove homogeneous coordinate
    transformed_matrix = transformed_homogeneous[..., :3]

    assert transformed_points.dtype == points.dtype
    assert transformed_points.shape == points.shape

    # Convert to numpy arrays and compare
    transformed_points_np = np.array(transformed_points)

    assert np.allclose(transformed_points_np, transformed_matrix, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_transform_points_scipy(backend, batch_dims, pass_xp):
    """Test point transformation against scipy spatial transform."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    pytest.importorskip("scipy.spatial")
    from scipy.spatial.transform import Rotation as R

    # Create random SE3 transformation
    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=32)

    # Generate random points
    points = random_points(batch_dims=batch_dims, n_points=10, backend=backend, precision=32)

    # Transform using nanomanifold
    transformed_points = SE3.transform_points(se3, points, **xp_kwargs)

    # Convert to numpy for scipy
    se3_np = np.array(se3)
    points_np = np.array(points)

    # Extract quaternion and translation
    q_np = se3_np[..., :4]
    t_np = se3_np[..., 4:7]

    # Convert from [w, x, y, z] to scipy's [x, y, z, w] format
    q_scipy = np.concatenate([q_np[..., 1:4], q_np[..., 0:1]], axis=-1)

    # Reshape for scipy processing
    q_flat = q_scipy.reshape(-1, 4)
    t_flat = t_np.reshape(-1, 3)
    points_flat = points_np.reshape(-1, points_np.shape[-2], 3)

    # Transform using scipy
    transformed_scipy = []
    for i in range(len(q_flat)):
        r = R.from_quat(q_flat[i : i + 1])
        # Apply rotation then translation: p' = R*p + t
        rotated = r.apply(points_flat[i])
        transformed = rotated + t_flat[i : i + 1, None, :]
        transformed_scipy.append(transformed)

    transformed_scipy = np.array(transformed_scipy).reshape(points_np.shape)

    # Convert to numpy for comparison
    transformed_points_np = np.array(transformed_points)

    assert np.allclose(transformed_points_np, transformed_scipy, atol=1e-6)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_transform_points_differentiability_torch(batch_dims):
    """Test that transform_points function is differentiable with PyTorch."""
    torch = pytest.importorskip("torch")
    # Use double precision for gradient checking as recommended by PyTorch
    dtype = torch.float64

    # Random SE3 input
    se3 = random_se3(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)
    points = random_points(batch_dims=batch_dims, n_points=5, backend="torch").to(dtype)

    # Check gradients of SE3.transform_points with respect to SE3 transformation
    def f(se3):
        return SE3.transform_points(se3, points)

    assert torch.autograd.gradcheck(f, (se3,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_transform_points_jittability_jax(batch_dims):
    """Test that transform_points function is JIT-compatible with JAX."""
    jax = pytest.importorskip("jax")

    # Create test SE3 transformation and points
    se3 = random_se3(batch_dims=batch_dims, backend="jax", precision=32)
    points = random_points(batch_dims=batch_dims, n_points=5, backend="jax", precision=32)

    # Define JIT-compiled function
    @jax.jit
    def jit_transform_points(se3, points):
        return SE3.transform_points(se3, points)

    # Test that JIT compilation works and compare with non-JIT
    result_jit = jit_transform_points(se3, points)
    result_non_jit = SE3.transform_points(se3, points)

    # Verify results match between JIT and non-JIT
    assert jax.numpy.allclose(result_jit, result_non_jit, atol=1e-6)

    # Verify result has correct shape
    expected_shape = batch_dims + (5, 3)
    assert result_jit.shape == expected_shape


def test_transform_points_specific_transformations():
    """Test transformation of specific known cases."""
    # Pure translation: [1, 0, 0, 0, 1, 2, 3]
    se3_translation = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # Origin and unit x

    transformed = SE3.transform_points(se3_translation, points)
    transformed_np = np.array(transformed)

    # Expected: points translated by (1, 2, 3)
    expected = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]])
    assert np.allclose(transformed_np, expected, atol=1e-6)

    # 90-degree rotation around z-axis with translation
    sqrt2_inv = 1.0 / np.sqrt(2.0)
    se3_rot_trans = np.array([sqrt2_inv, 0.0, 0.0, sqrt2_inv, 1.0, 0.0, 0.0])
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # Unit x and unit y

    transformed = SE3.transform_points(se3_rot_trans, points)
    transformed_np = np.array(transformed)

    # Expected: 90° rotation around z + translation (1,0,0)
    # (1,0,0) -> (0,1,0) + (1,0,0) = (1,1,0)
    # (0,1,0) -> (-1,0,0) + (1,0,0) = (0,0,0)
    expected = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(transformed_np, expected, atol=1e-6)


def test_transform_points_composition():
    """Test that composing transformations gives same result as composed transformation."""
    # Create two random SE3 transformations
    se3_1 = random_se3(batch_dims=(), backend="numpy", precision=32)
    se3_2 = random_se3(batch_dims=(), backend="numpy", precision=32)

    # Compose them
    se3_composed = SE3.multiply(se3_1, se3_2)

    # Create random points
    points = random_points(batch_dims=(), n_points=5, backend="numpy", precision=32)

    # Method 1: Transform with composed transformation
    result_composed = SE3.transform_points(se3_composed, points)

    # Method 2: Transform sequentially (se3_2 first, then se3_1)
    intermediate = SE3.transform_points(se3_2, points)
    result_sequential = SE3.transform_points(se3_1, intermediate)

    # Results should match
    result_composed_np = np.array(result_composed)
    result_sequential_np = np.array(result_sequential)

    assert np.allclose(result_composed_np, result_sequential_np, atol=1e-6)


def test_transform_points_pure_rotation():
    """Test that SE3 with zero translation matches SO3 rotation."""

    # Create random quaternion
    quat = np.array([0.707, 0.707, 0.0, 0.0])  # 90° around x-axis
    quat = quat / np.linalg.norm(quat)  # Normalize

    # Create SE3 with zero translation
    se3_pure_rotation = np.concatenate([quat, np.zeros(3)])

    # Create test points
    points = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # Unit y and unit z

    # Transform using SE3
    se3_result = SE3.transform_points(se3_pure_rotation, points)

    # Transform using SO3
    so3_result = SO3.rotate_points(quat, points)

    # Results should match
    se3_result_np = np.array(se3_result)
    so3_result_np = np.array(so3_result)

    assert np.allclose(se3_result_np, so3_result_np, atol=1e-6)


def test_transform_points_broadcasting():
    """Test broadcasting between different SE3 and points batch sizes."""
    # Single SE3 transformation and batch of points
    se3_single = random_se3(batch_dims=(), backend="numpy", precision=32)
    points_batch = random_points(batch_dims=(3,), n_points=5, backend="numpy", precision=32)

    # Test broadcasting: single SE3 transforms multiple point sets
    result = SE3.transform_points(se3_single, points_batch)
    assert result.shape == (3, 5, 3)

    # Batch of SE3 transformations and single point set
    se3_batch = random_se3(batch_dims=(3,), backend="numpy", precision=32)
    points_single = random_points(batch_dims=(), n_points=5, backend="numpy", precision=32)

    # Test broadcasting: multiple SE3 transforms single point set
    result = SE3.transform_points(se3_batch, points_single)
    assert result.shape == (3, 5, 3)

    # Verify results are valid (finite numbers)
    result_np = np.array(result)
    assert np.all(np.isfinite(result_np))


def test_transform_points_single_point():
    """Test transformation of single point (edge case)."""
    # Create SE3 transformation
    se3 = np.array([0.707, 0.0, 0.707, 0.0, 1.0, 2.0, 3.0])  # 90° around y + translation

    # Single point
    point = np.array([[1.0, 0.0, 0.0]])  # Shape (1, 3)

    # Transform
    transformed = SE3.transform_points(se3, point)
    transformed_np = np.array(transformed)

    # Expected: 90° rotation around y transforms (1,0,0) -> (0,0,-1), then add (1,2,3)
    expected = np.array([[1.0, 2.0, 2.0]])
    assert transformed.shape == (1, 3)
    assert np.allclose(transformed_np, expected, atol=1e-6)
