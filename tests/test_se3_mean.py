import numpy as np
import pytest
from conftest import ATOL, TEST_BATCH_DIMS, TEST_PRECISIONS, awkward_quaternions, random_se3

from nanomanifold import SE3


@pytest.mark.parametrize("copies", [lambda T: [T], lambda T: [T, T, T]], ids=["single", "identical"])
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_mean_of_one_transform_is_that_transform(copies, precision):
    quat = awkward_quaternions(precision)
    se3 = np.concatenate([quat, np.random.default_rng(0).normal(size=(len(quat), 3)).astype(quat.dtype)], axis=-1)

    result = SE3.mean(copies(se3))

    assert result.dtype == se3.dtype
    assert result.shape == se3.shape
    assert np.allclose(np.abs(np.sum(se3[..., :4] * result[..., :4], axis=-1)), 1.0, atol=ATOL[precision])
    assert np.allclose(se3[..., 4:], result[..., 4:], atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_weighted_mean_single_weight(batch_dims, precision):
    """Weighted mean with weight [1, 0] should return the first transform exactly."""
    xp = np

    se3_1 = random_se3(batch_dims=batch_dims, precision=precision)
    se3_2 = random_se3(batch_dims=batch_dims, precision=precision)

    weights = xp.asarray(np.array([1.0, 0.0]).astype(f"float{precision}"))
    weights = xp.broadcast_to(weights, batch_dims + (2,))

    result = SE3.weighted_mean([se3_1, se3_2], weights)

    se3_1_np = np.array(se3_1)
    result_np = np.array(result)

    dot = np.sum(se3_1_np[..., :4] * result_np[..., :4], axis=-1)
    assert np.allclose(np.abs(dot), 1.0, atol=ATOL[precision])
    assert np.allclose(se3_1_np[..., 4:7], result_np[..., 4:7], atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_mean_translation_correctness(batch_dims, precision):
    """Mean translation should be the average of the input translations."""

    se3_1 = random_se3(batch_dims=batch_dims, precision=precision)
    se3_2 = random_se3(batch_dims=batch_dims, precision=precision)

    result = SE3.mean([se3_1, se3_2])

    se3_1_np = np.array(se3_1)
    se3_2_np = np.array(se3_2)
    result_np = np.array(result)

    expected_t = (se3_1_np[..., 4:7] + se3_2_np[..., 4:7]) / 2.0
    assert np.allclose(result_np[..., 4:7], expected_t, atol=ATOL[precision])


def test_mean_empty_raises():
    """Mean of empty sequence should raise ValueError."""
    with pytest.raises(ValueError, match="Cannot compute mean"):
        SE3.mean([])
