import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PASS_XP, TEST_PRECISIONS, get_xp_kwargs, random_se3

from nanomanifold import SE3


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
def test_slerp_endpoints(backend, batch_dims, precision, pass_xp):
    """Test that slerp(T1, T2, [0]) ≈ T1 and slerp(T1, T2, [1]) ≈ T2."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend)
    dtype = get_dtype(backend, precision)

    se3_1 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)
    se3_2 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    t_0 = xp.asarray([0.0], dtype=dtype)
    t_1 = xp.asarray([1.0], dtype=dtype)

    result_0 = SE3.slerp(se3_1, se3_2, t_0, **xp_kwargs)
    result_1 = SE3.slerp(se3_1, se3_2, t_1, **xp_kwargs)

    assert result_0.shape == batch_dims + (1, 7)
    assert result_1.shape == batch_dims + (1, 7)

    se3_1_np = np.array(se3_1)
    se3_2_np = np.array(se3_2)
    result_0_np = np.array(result_0).squeeze(-2)
    result_1_np = np.array(result_1).squeeze(-2)

    # Check quaternion equivalence
    dot_0 = np.sum(se3_1_np[..., :4] * result_0_np[..., :4], axis=-1)
    assert np.allclose(np.abs(dot_0), 1.0, atol=ATOL[precision])
    # Check translation
    assert np.allclose(se3_1_np[..., 4:7], result_0_np[..., 4:7], atol=ATOL[precision])

    dot_1 = np.sum(se3_2_np[..., :4] * result_1_np[..., :4], axis=-1)
    assert np.allclose(np.abs(dot_1), 1.0, atol=ATOL[precision])
    assert np.allclose(se3_2_np[..., 4:7], result_1_np[..., 4:7], atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_slerp_midpoint_translation(backend, batch_dims, precision, pass_xp):
    """Test that the midpoint translation is the average of endpoints."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend)
    dtype = get_dtype(backend, precision)

    se3_1 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)
    se3_2 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    t_half = xp.asarray([0.5], dtype=dtype)
    result = SE3.slerp(se3_1, se3_2, t_half, **xp_kwargs)

    se3_1_np = np.array(se3_1)
    se3_2_np = np.array(se3_2)
    result_np = np.array(result).squeeze(-2)

    expected_t = (se3_1_np[..., 4:7] + se3_2_np[..., 4:7]) / 2.0
    assert np.allclose(result_np[..., 4:7], expected_t, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_slerp_same_transform(backend, batch_dims, precision, pass_xp):
    """Slerping between the same transform should return that transform."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend)
    dtype = get_dtype(backend, precision)

    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)
    t_vals = xp.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=dtype)

    result = SE3.slerp(se3, se3, t_vals, **xp_kwargs)

    assert result.shape == batch_dims + (5, 7)

    se3_np = np.array(se3)
    result_np = np.array(result)

    for i in range(5):
        r = result_np[..., i, :]
        dot = np.sum(se3_np[..., :4] * r[..., :4], axis=-1)
        assert np.allclose(np.abs(dot), 1.0, atol=ATOL[precision])
        assert np.allclose(se3_np[..., 4:7], r[..., 4:7], atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_slerp_differentiability_torch(batch_dims):
    """Test that slerp is differentiable with PyTorch."""
    torch = pytest.importorskip("torch")
    dtype = torch.float64

    se3_1 = random_se3(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)
    se3_2 = random_se3(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)
    t = torch.tensor([0.3], dtype=dtype)

    def f(a, b):
        return SE3.slerp(a, b, t).sum()

    assert torch.autograd.gradcheck(f, (se3_1, se3_2), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_slerp_jittability_jax(batch_dims):
    """Test that slerp is JIT-compatible with JAX."""
    jax = pytest.importorskip("jax")
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name("jax")

    se3_1 = random_se3(batch_dims=batch_dims, backend="jax", precision=32)
    se3_2 = random_se3(batch_dims=batch_dims, backend="jax", precision=32)
    t = xp.asarray([0.5])

    @jax.jit
    def jit_slerp(a, b, t):
        return SE3.slerp(a, b, t)

    result_jit = jit_slerp(se3_1, se3_2, t)
    result_non_jit = SE3.slerp(se3_1, se3_2, t)

    assert jax.numpy.allclose(result_jit, result_non_jit, atol=1e-6)
