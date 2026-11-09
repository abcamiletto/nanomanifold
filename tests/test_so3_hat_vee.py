import numpy as np
import pytest
from conftest import ATOL, TEST_BATCH_DIMS, TEST_PRECISIONS

from nanomanifold import SO3


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_hat_skew_symmetric(batch_dims, precision):
    """Test that hat produces skew-symmetric matrices."""
    xp = np
    dtype = np.dtype(f"float{precision}")

    # Create random tangent vectors
    w = 0.1 * xp.asarray(np.random.randn(*batch_dims, 3), dtype=dtype)
    W = SO3.hat(w)

    assert W.shape == batch_dims + (3, 3)
    assert W.dtype == w.dtype

    W_np = np.array(W)
    # Properly transpose batch dimensions: swap last two axes
    W_T_np = np.transpose(W_np, axes=tuple(range(len(W.shape) - 2)) + (-1, -2))

    # Check that W = -W^T (skew-symmetric property)
    assert np.allclose(W_np, -W_T_np, atol=ATOL[precision])

    # Check that diagonal is zero
    diag_indices = np.arange(3)
    diagonal = W_np[..., diag_indices, diag_indices]
    assert np.allclose(diagonal, 0.0, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_hat_zero_vector(batch_dims, precision):
    """Test hat operator on zero vector gives zero matrix."""
    xp = np
    dtype = np.dtype(f"float{precision}")

    w = xp.zeros(batch_dims + (3,), dtype=dtype)
    W = SO3.hat(w)

    assert W.shape == batch_dims + (3, 3)
    assert W.dtype == w.dtype

    W_np = np.array(W)
    assert np.allclose(W_np, 0.0, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_vee_hat_inverse(batch_dims, precision):
    """Test that vee(hat(w)) = w."""
    xp = np
    dtype = np.dtype(f"float{precision}")

    # Create random tangent vectors
    w = 0.1 * xp.asarray(np.random.randn(*batch_dims, 3), dtype=dtype)
    w_recovered = SO3.vee(SO3.hat(w))

    assert w_recovered.shape == batch_dims + (3,)
    assert w_recovered.dtype == w.dtype

    w_np = np.array(w)
    w_recovered_np = np.array(w_recovered)
    assert np.allclose(w_np, w_recovered_np, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_hat_vee_inverse_skew_symmetric(batch_dims, precision):
    """Test that hat(vee(W)) = W for skew-symmetric W."""
    xp = np
    dtype = np.dtype(f"float{precision}")

    # Create random tangent vector, then convert to skew-symmetric matrix
    w = 0.1 * xp.asarray(np.random.randn(*batch_dims, 3), dtype=dtype)
    W = SO3.hat(w)
    W_recovered = SO3.hat(SO3.vee(W))

    assert W_recovered.shape == batch_dims + (3, 3)
    assert W_recovered.dtype == W.dtype

    W_np = np.array(W)
    W_recovered_np = np.array(W_recovered)
    assert np.allclose(W_np, W_recovered_np, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_vee_zero_matrix(batch_dims, precision):
    """Test vee operator on zero matrix gives zero vector."""
    xp = np
    dtype = np.dtype(f"float{precision}")

    W = xp.zeros(batch_dims + (3, 3), dtype=dtype)
    w = SO3.vee(W)

    assert w.shape == batch_dims + (3,)
    assert w.dtype == W.dtype

    w_np = np.array(w)
    assert np.allclose(w_np, 0.0, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_hat_vee_numerical_precision(batch_dims, precision):
    """Test numerical precision of hat/vee operations."""
    xp = np
    dtype = np.dtype(f"float{precision}")

    if precision == 16:
        atol = 1e-3
    elif precision == 32:
        atol = 1e-6
    else:
        atol = 1e-14

    # Test with mathematically meaningful values
    pi = xp.asarray(np.pi, dtype=dtype)
    e = xp.asarray(np.e, dtype=dtype)
    sqrt2 = xp.asarray(float(np.sqrt(2)), dtype=dtype)

    if len(batch_dims) == 0:
        w = xp.stack([pi, e, sqrt2])
    else:
        # For batch dimensions, create appropriate shape
        w_base = xp.stack([pi, e, sqrt2])
        w = xp.broadcast_to(w_base, batch_dims + (3,))

    w_recovered = SO3.vee(SO3.hat(w))

    assert w_recovered.dtype == w.dtype

    if precision >= 16:
        w_np = np.array(w)
        w_recovered_np = np.array(w_recovered)
        assert np.allclose(w_np, w_recovered_np, atol=atol)


def test_hat_specific_values():
    """Test hat operator with specific known values."""
    w = np.array([1.0, 2.0, 3.0])
    W = SO3.hat(w)

    expected = np.array([[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]])

    W_np = np.array(W)
    assert np.allclose(W_np, expected)


def test_vee_specific_values():
    """Test vee operator with specific known values."""
    W = np.array([[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]])
    w = SO3.vee(W)

    expected = np.array([1.0, 2.0, 3.0])

    w_np = np.array(w)
    assert np.allclose(w_np, expected)


def test_hat_vee_large_values():
    """Test hat and vee with large values."""
    xp = np

    w = xp.asarray([1e6, -1e6, 1e6], dtype=xp.float32)
    W = SO3.hat(w)

    assert W.dtype == w.dtype

    # Should still be skew-symmetric
    W_np = np.array(W)
    W_T_np = np.transpose(W_np, axes=(-1, -2))  # Just swap last two axes
    assert np.allclose(W_np, -W_T_np)


def test_hat_vee_small_values():
    """Test hat and vee with very small values."""
    xp = np

    w = xp.asarray([1e-10, -1e-10, 1e-10], dtype=xp.float32)
    W = SO3.hat(w)

    assert W.dtype == w.dtype

    # Should still be skew-symmetric
    W_np = np.array(W)
    W_T_np = np.transpose(W_np, axes=(-1, -2))  # Just swap last two axes
    assert np.allclose(W_np, -W_T_np, atol=1e-15)
