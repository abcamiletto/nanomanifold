import numpy as np
import pytest
from conftest import ATOL, TEST_BATCH_DIMS, TEST_PRECISIONS

from nanomanifold import SE3, SO3


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_hat_correct_structure(batch_dims, precision):
    """Test that hat produces correct 4x4 structure: top-left is SO3.hat, top-right is rho, bottom row is zeros."""
    xp = np
    dtype = np.dtype(f"float{precision}")

    v = 0.1 * xp.asarray(np.random.randn(*batch_dims, 6), dtype=dtype)
    M = SE3.hat(v)

    assert M.shape == batch_dims + (4, 4)
    assert M.dtype == v.dtype

    M_np = np.array(M)

    # Bottom row should be zeros
    assert np.allclose(M_np[..., 3, :], 0.0, atol=ATOL[precision])

    # Top-left 3x3 should match SO3.hat(omega)
    omega = v[..., :3]
    so3_hat_result = SO3.hat(omega)
    so3_hat_np = np.array(so3_hat_result)
    assert np.allclose(M_np[..., :3, :3], so3_hat_np, atol=ATOL[precision])

    # Top-right column should be rho
    rho_np = np.array(v[..., 3:6])
    assert np.allclose(M_np[..., :3, 3], rho_np, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_hat_zero_input(batch_dims, precision):
    """Test hat on zero vector gives zero matrix."""
    xp = np
    dtype = np.dtype(f"float{precision}")

    v = xp.zeros(batch_dims + (6,), dtype=dtype)
    M = SE3.hat(v)

    assert M.shape == batch_dims + (4, 4)
    M_np = np.array(M)
    assert np.allclose(M_np, 0.0, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_vee_hat_inverse(batch_dims, precision):
    """Test that vee(hat(v)) == v."""
    xp = np
    dtype = np.dtype(f"float{precision}")

    v = 0.1 * xp.asarray(np.random.randn(*batch_dims, 6), dtype=dtype)
    v_recovered = SE3.vee(SE3.hat(v))

    assert v_recovered.shape == batch_dims + (6,)
    assert v_recovered.dtype == v.dtype

    v_np = np.array(v)
    v_recovered_np = np.array(v_recovered)
    assert np.allclose(v_np, v_recovered_np, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_hat_vee_inverse(batch_dims, precision):
    """Test that hat(vee(M)) == M for valid se(3) matrices."""
    xp = np
    dtype = np.dtype(f"float{precision}")

    v = 0.1 * xp.asarray(np.random.randn(*batch_dims, 6), dtype=dtype)
    M = SE3.hat(v)
    M_recovered = SE3.hat(SE3.vee(M))

    assert M_recovered.shape == batch_dims + (4, 4)
    assert M_recovered.dtype == M.dtype

    M_np = np.array(M)
    M_recovered_np = np.array(M_recovered)
    assert np.allclose(M_np, M_recovered_np, atol=ATOL[precision])


def test_hat_specific_values():
    """Test hat with specific known values."""
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    M = SE3.hat(v)
    M_np = np.array(M)

    expected = np.array(
        [
            [0.0, -3.0, 2.0, 4.0],
            [3.0, 0.0, -1.0, 5.0],
            [-2.0, 1.0, 0.0, 6.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert np.allclose(M_np, expected)


def test_vee_specific_values():
    """Test vee with specific known values."""
    M = np.array(
        [
            [0.0, -3.0, 2.0, 4.0],
            [3.0, 0.0, -1.0, 5.0],
            [-2.0, 1.0, 0.0, 6.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    v = SE3.vee(M)
    v_np = np.array(v)

    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert np.allclose(v_np, expected)
