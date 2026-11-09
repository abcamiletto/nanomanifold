import numpy as np
import pytest
from conftest import ATOL, TEST_BATCH_DIMS, TEST_PRECISIONS, random_se3

from nanomanifold import SE3


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_canonicalize_w_nonnegative(batch_dims, precision):
    """Test that canonicalize ensures w >= 0."""
    se3 = random_se3(batch_dims=batch_dims, precision=precision)

    result = SE3.canonicalize(se3)

    assert result.dtype == se3.dtype
    assert result.shape == se3.shape

    result_np = np.array(result)
    assert np.all(result_np[..., 0] >= -ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_canonicalize_unit_quaternion(batch_dims, precision):
    """Test that canonicalize preserves unit quaternion norm."""
    se3 = random_se3(batch_dims=batch_dims, precision=precision)

    result = SE3.canonicalize(se3)
    result_np = np.array(result)

    quat_norms = np.linalg.norm(result_np[..., :4], axis=-1)
    assert np.allclose(quat_norms, 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_canonicalize_preserves_translation(batch_dims, precision):
    """Test that canonicalize does not modify the translation part."""
    se3 = random_se3(batch_dims=batch_dims, precision=precision)

    result = SE3.canonicalize(se3)

    se3_np = np.array(se3)
    result_np = np.array(result)

    assert np.allclose(se3_np[..., 4:], result_np[..., 4:], atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_canonicalize_negated_quaternion(batch_dims, precision):
    """Test that canonicalize(se3) and canonicalize(-q, t) give the same result."""
    xp = np

    se3 = random_se3(batch_dims=batch_dims, precision=precision)

    # Negate only the quaternion part
    quat_neg = -se3[..., :4]
    translation = se3[..., 4:]
    se3_neg = xp.concatenate([quat_neg, translation], axis=-1)

    result1 = SE3.canonicalize(se3)
    result2 = SE3.canonicalize(se3_neg)

    result1_np = np.array(result1)
    result2_np = np.array(result2)

    assert np.allclose(result1_np, result2_np, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_canonicalize_idempotent(batch_dims, precision):
    """Test that applying canonicalize twice gives the same result."""
    se3 = random_se3(batch_dims=batch_dims, precision=precision)

    result1 = SE3.canonicalize(se3)
    result2 = SE3.canonicalize(result1)

    result1_np = np.array(result1)
    result2_np = np.array(result2)

    assert np.allclose(result1_np, result2_np, atol=ATOL[precision])
