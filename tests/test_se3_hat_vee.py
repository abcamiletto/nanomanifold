import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PASS_XP, TEST_PRECISIONS, get_xp_kwargs

from nanomanifold import SE3, SO3


def get_dtype(backend, precision):
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend)
    if precision == 16:
        return xp.float16
    elif precision == 32:
        return xp.float32
    else:
        return xp.float64


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_hat_correct_structure(backend, batch_dims, precision, pass_xp):
    """Test that hat produces correct 4x4 structure: top-left is SO3.hat, top-right is rho, bottom row is zeros."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend)
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    dtype = get_dtype(backend, precision)

    v = 0.1 * xp.asarray(np.random.randn(*batch_dims, 6), dtype=dtype)
    M = SE3.hat(v, **xp_kwargs)

    assert M.shape == batch_dims + (4, 4)
    assert M.dtype == v.dtype

    M_np = np.array(M)

    # Bottom row should be zeros
    assert np.allclose(M_np[..., 3, :], 0.0, atol=ATOL[precision])

    # Top-left 3x3 should match SO3.hat(omega)
    omega = v[..., :3]
    so3_hat_result = SO3.hat(omega, **xp_kwargs)
    so3_hat_np = np.array(so3_hat_result)
    assert np.allclose(M_np[..., :3, :3], so3_hat_np, atol=ATOL[precision])

    # Top-right column should be rho
    rho_np = np.array(v[..., 3:6])
    assert np.allclose(M_np[..., :3, 3], rho_np, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_hat_zero_input(backend, batch_dims, precision, pass_xp):
    """Test hat on zero vector gives zero matrix."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend)
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    dtype = get_dtype(backend, precision)

    v = xp.zeros(batch_dims + (6,), dtype=dtype)
    M = SE3.hat(v, **xp_kwargs)

    assert M.shape == batch_dims + (4, 4)
    M_np = np.array(M)
    assert np.allclose(M_np, 0.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_vee_hat_inverse(backend, batch_dims, precision, pass_xp):
    """Test that vee(hat(v)) == v."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend)
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    dtype = get_dtype(backend, precision)

    v = 0.1 * xp.asarray(np.random.randn(*batch_dims, 6), dtype=dtype)
    v_recovered = SE3.vee(SE3.hat(v, **xp_kwargs), **xp_kwargs)

    assert v_recovered.shape == batch_dims + (6,)
    assert v_recovered.dtype == v.dtype

    v_np = np.array(v)
    v_recovered_np = np.array(v_recovered)
    assert np.allclose(v_np, v_recovered_np, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_hat_vee_inverse(backend, batch_dims, precision, pass_xp):
    """Test that hat(vee(M)) == M for valid se(3) matrices."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend)
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    dtype = get_dtype(backend, precision)

    v = 0.1 * xp.asarray(np.random.randn(*batch_dims, 6), dtype=dtype)
    M = SE3.hat(v, **xp_kwargs)
    M_recovered = SE3.hat(SE3.vee(M, **xp_kwargs), **xp_kwargs)

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


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_hat_differentiability_torch(batch_dims):
    """Test that hat function is differentiable with PyTorch."""
    torch = pytest.importorskip("torch")
    dtype = torch.float64

    v = torch.randn(*batch_dims, 6, dtype=dtype, requires_grad=True)

    def f(v):
        return SE3.hat(v).sum()

    assert torch.autograd.gradcheck(f, (v,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_vee_differentiability_torch(batch_dims):
    """Test that vee function is differentiable with PyTorch."""
    torch = pytest.importorskip("torch")
    dtype = torch.float64

    # Create a valid se(3) matrix
    v = torch.randn(*batch_dims, 6, dtype=dtype)
    M = SE3.hat(v).detach().requires_grad_(True)

    def f(M):
        return SE3.vee(M).sum()

    assert torch.autograd.gradcheck(f, (M,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_hat_vee_jittability_jax(batch_dims):
    """Test that hat and vee are JIT-compatible with JAX."""
    jax = pytest.importorskip("jax")
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name("jax")

    v = xp.asarray(np.random.randn(*batch_dims, 6).astype("float32"))

    @jax.jit
    def jit_hat(v):
        return SE3.hat(v)

    @jax.jit
    def jit_vee(M):
        return SE3.vee(M)

    M_jit = jit_hat(v)
    v_jit = jit_vee(M_jit)

    M_non_jit = SE3.hat(v)
    v_non_jit = SE3.vee(M_non_jit)

    assert jax.numpy.allclose(M_jit, M_non_jit, atol=1e-6)
    assert jax.numpy.allclose(v_jit, v_non_jit, atol=1e-6)
