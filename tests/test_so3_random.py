"""Tests for SO3.random uniform sampling."""

import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PASS_XP, TEST_PRECISIONS, get_xp_kwargs

from nanomanifold import SO3
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
    """Output shape is (*shape, 4)."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    dtype = _get_dtype(backend, precision, pass_xp)

    if backend == "jax" and pass_xp:
        jax = pytest.importorskip("jax")
        xp_kwargs["key"] = jax.random.PRNGKey(0)
    elif backend == "jax" and not pass_xp:
        pytest.skip("random() without xp defaults to numpy, not jax")

    q = SO3.random(*batch_dims, dtype=dtype, **xp_kwargs)
    assert q.shape == batch_dims + (4,)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_random_unit_norm(backend, batch_dims, precision, pass_xp):
    """All quaternions have unit norm."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    dtype = _get_dtype(backend, precision, pass_xp)

    if backend == "jax" and pass_xp:
        jax = pytest.importorskip("jax")
        xp_kwargs["key"] = jax.random.PRNGKey(1)
    elif backend == "jax" and not pass_xp:
        pytest.skip("random() without xp defaults to numpy, not jax")

    q = SO3.random(*batch_dims, dtype=dtype, **xp_kwargs)
    q_np = np.array(q)
    norms = np.linalg.norm(q_np, axis=-1)
    assert np.allclose(norms, 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_random_canonical(backend, batch_dims, precision, pass_xp):
    """All quaternions have w >= 0."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    dtype = _get_dtype(backend, precision, pass_xp)

    if backend == "jax" and pass_xp:
        jax = pytest.importorskip("jax")
        xp_kwargs["key"] = jax.random.PRNGKey(2)
    elif backend == "jax" and not pass_xp:
        pytest.skip("random() without xp defaults to numpy, not jax")

    q = SO3.random(*batch_dims, dtype=dtype, **xp_kwargs)
    q_np = np.array(q)
    assert np.all(q_np[..., 0] >= -1e-7)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_random_statistical_uniformity(backend):
    """For large N, mean dot product with any fixed quaternion ~ 0."""
    N = 10000
    dtype = _get_dtype(backend, 32)
    xp_kwargs = {}
    if backend != "numpy":
        xp = get_namespace_by_name(backend)
        xp_kwargs["xp"] = xp
    if backend == "jax":
        jax = pytest.importorskip("jax")
        xp_kwargs["key"] = jax.random.PRNGKey(42)

    q = SO3.random(N, dtype=dtype, **xp_kwargs)
    q_np = np.array(q)

    # Use xyz components (not w) to test uniformity; w is biased by canonicalization
    ref = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    dots = np.sum(q_np * ref, axis=-1)
    mean_dot = np.mean(dots)
    # For uniform SO(3) sampling, mean of x component ~ 0
    assert abs(mean_dot) < 0.05, f"Mean dot product {mean_dot} too far from 0"


def test_random_jax_deterministic():
    """Same JAX key gives identical output."""
    jax = pytest.importorskip("jax")
    xp = get_namespace_by_name("jax")
    key = jax.random.PRNGKey(123)

    q1 = SO3.random(5, xp=xp, key=key)
    q2 = SO3.random(5, xp=xp, key=key)
    np.testing.assert_array_equal(np.array(q1), np.array(q2))


def test_random_jax_requires_key():
    """JAX backend raises ValueError without key."""
    pytest.importorskip("jax")
    xp = get_namespace_by_name("jax")
    with pytest.raises(ValueError, match="PRNG key"):
        SO3.random(5, xp=xp)


def test_random_jax_jit():
    """SO3.random works under jax.jit."""
    jax = pytest.importorskip("jax")
    xp = get_namespace_by_name("jax")

    @jax.jit
    def sample(key):
        return SO3.random(5, xp=xp, key=key)

    key = jax.random.PRNGKey(0)
    q = sample(key)
    assert q.shape == (5, 4)
    norms = np.linalg.norm(np.array(q), axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-6)
