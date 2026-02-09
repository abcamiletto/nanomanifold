"""Tests for SE3.random uniform sampling."""

import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PASS_XP, TEST_PRECISIONS, get_xp_kwargs

from nanomanifold import SE3
from nanomanifold.common import get_namespace_by_name


def _get_dtype(backend, precision, pass_xp=True):
    # When xp is not passed, random() defaults to numpy, so use numpy dtypes
    if not pass_xp or backend == "numpy":
        return getattr(np, f"float{precision}")
    elif backend == "torch":
        torch = pytest.importorskip("torch")
        return getattr(torch, f"float{precision}")
    else:
        jax = pytest.importorskip("jax")
        return getattr(jax.numpy, f"float{precision}")


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_random_shape(backend, batch_dims, precision, pass_xp):
    """Output shape is (*shape, 7)."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    dtype = _get_dtype(backend, precision, pass_xp)

    if backend == "jax" and pass_xp:
        jax = pytest.importorskip("jax")
        xp_kwargs["key"] = jax.random.PRNGKey(0)
    elif backend == "jax" and not pass_xp:
        pytest.skip("random() without xp defaults to numpy, not jax")

    se3 = SE3.random(*batch_dims, dtype=dtype, **xp_kwargs)
    assert se3.shape == batch_dims + (7,)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_random_rotation_unit_norm(backend, batch_dims, precision, pass_xp):
    """Rotation part has unit norm."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    dtype = _get_dtype(backend, precision, pass_xp)

    if backend == "jax" and pass_xp:
        jax = pytest.importorskip("jax")
        xp_kwargs["key"] = jax.random.PRNGKey(1)
    elif backend == "jax" and not pass_xp:
        pytest.skip("random() without xp defaults to numpy, not jax")

    se3 = SE3.random(*batch_dims, dtype=dtype, **xp_kwargs)
    se3_np = np.array(se3)
    q_norms = np.linalg.norm(se3_np[..., :4], axis=-1)
    assert np.allclose(q_norms, 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_random_rotation_canonical(backend, batch_dims, precision, pass_xp):
    """Rotation part has w >= 0."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    dtype = _get_dtype(backend, precision, pass_xp)

    if backend == "jax" and pass_xp:
        jax = pytest.importorskip("jax")
        xp_kwargs["key"] = jax.random.PRNGKey(2)
    elif backend == "jax" and not pass_xp:
        pytest.skip("random() without xp defaults to numpy, not jax")

    se3 = SE3.random(*batch_dims, dtype=dtype, **xp_kwargs)
    se3_np = np.array(se3)
    assert np.all(se3_np[..., 0] >= -1e-7)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_random_translation_range(backend, batch_dims, precision, pass_xp):
    """Translation values are in [-1, 1]."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    dtype = _get_dtype(backend, precision, pass_xp)

    if backend == "jax" and pass_xp:
        jax = pytest.importorskip("jax")
        xp_kwargs["key"] = jax.random.PRNGKey(3)
    elif backend == "jax" and not pass_xp:
        pytest.skip("random() without xp defaults to numpy, not jax")

    se3 = SE3.random(*batch_dims, dtype=dtype, **xp_kwargs)
    se3_np = np.array(se3)
    t = se3_np[..., 4:]
    assert np.all(t >= -1.0 - 1e-7)
    assert np.all(t <= 1.0 + 1e-7)


def test_random_jax_deterministic():
    """Same JAX key gives identical output."""
    jax = pytest.importorskip("jax")
    xp = get_namespace_by_name("jax")
    key = jax.random.PRNGKey(123)

    se3_1 = SE3.random(5, xp=xp, key=key)
    se3_2 = SE3.random(5, xp=xp, key=key)
    np.testing.assert_array_equal(np.array(se3_1), np.array(se3_2))


def test_random_jax_requires_key():
    """JAX backend raises ValueError without key."""
    pytest.importorskip("jax")
    xp = get_namespace_by_name("jax")
    with pytest.raises(ValueError, match="PRNG key"):
        SE3.random(5, xp=xp)


def test_random_jax_jit():
    """SE3.random works under jax.jit."""
    jax = pytest.importorskip("jax")
    xp = get_namespace_by_name("jax")

    @jax.jit
    def sample(key):
        return SE3.random(5, xp=xp, key=key)

    key = jax.random.PRNGKey(0)
    se3 = sample(key)
    assert se3.shape == (5, 7)
    se3_np = np.array(se3)
    q_norms = np.linalg.norm(se3_np[..., :4], axis=-1)
    assert np.allclose(q_norms, 1.0, atol=1e-6)
