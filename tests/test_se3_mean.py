import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PASS_XP, TEST_PRECISIONS, get_xp_kwargs, random_se3

from nanomanifold import SE3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_mean_single_transform(backend, batch_dims, precision, pass_xp):
    """Mean of a single transform should return that transform."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    result = SE3.mean([se3], **xp_kwargs)

    assert result.shape == se3.shape

    se3_np = np.array(se3)
    result_np = np.array(result)

    # Check quaternion equivalence
    dot = np.sum(se3_np[..., :4] * result_np[..., :4], axis=-1)
    assert np.allclose(np.abs(dot), 1.0, atol=ATOL[precision])
    # Check translation
    assert np.allclose(se3_np[..., 4:7], result_np[..., 4:7], atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_mean_identical_transforms(backend, batch_dims, precision, pass_xp):
    """Mean of identical transforms should return that transform."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    result = SE3.mean([se3, se3, se3], **xp_kwargs)

    assert result.shape == se3.shape

    se3_np = np.array(se3)
    result_np = np.array(result)

    dot = np.sum(se3_np[..., :4] * result_np[..., :4], axis=-1)
    assert np.allclose(np.abs(dot), 1.0, atol=ATOL[precision])
    assert np.allclose(se3_np[..., 4:7], result_np[..., 4:7], atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_weighted_mean_single_weight(backend, batch_dims, precision, pass_xp):
    """Weighted mean with weight [1, 0] should return the first transform exactly."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend)

    se3_1 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)
    se3_2 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    weights = xp.asarray(np.array([1.0, 0.0]).astype(f"float{precision}"))
    weights = xp.broadcast_to(weights, batch_dims + (2,))

    result = SE3.weighted_mean([se3_1, se3_2], weights, **xp_kwargs)

    se3_1_np = np.array(se3_1)
    result_np = np.array(result)

    dot = np.sum(se3_1_np[..., :4] * result_np[..., :4], axis=-1)
    assert np.allclose(np.abs(dot), 1.0, atol=ATOL[precision])
    assert np.allclose(se3_1_np[..., 4:7], result_np[..., 4:7], atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_mean_translation_correctness(backend, batch_dims, precision, pass_xp):
    """Mean translation should be the average of the input translations."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)

    se3_1 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)
    se3_2 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    result = SE3.mean([se3_1, se3_2], **xp_kwargs)

    se3_1_np = np.array(se3_1)
    se3_2_np = np.array(se3_2)
    result_np = np.array(result)

    expected_t = (se3_1_np[..., 4:7] + se3_2_np[..., 4:7]) / 2.0
    assert np.allclose(result_np[..., 4:7], expected_t, atol=ATOL[precision])


def test_mean_empty_raises():
    """Mean of empty sequence should raise ValueError."""
    with pytest.raises(ValueError, match="Cannot compute mean"):
        SE3.mean([])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_mean_jittability_jax(batch_dims):
    """Test that mean is JIT-compatible with JAX."""
    jax = pytest.importorskip("jax")

    se3_1 = random_se3(batch_dims=batch_dims, backend="jax", precision=32)
    se3_2 = random_se3(batch_dims=batch_dims, backend="jax", precision=32)

    @jax.jit
    def jit_mean(a, b):
        return SE3.mean([a, b])

    result_jit = jit_mean(se3_1, se3_2)
    result_non_jit = SE3.mean([se3_1, se3_2])

    assert jax.numpy.allclose(result_jit, result_non_jit, atol=1e-6)
