import numpy as np
import pytest
from conftest import ATOL, TEST_BATCH_DIMS, TEST_PRECISIONS, random_se3

from nanomanifold import SE3


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_slerp_endpoints(batch_dims, precision):
    """Test that slerp(T1, T2, [0]) ≈ T1 and slerp(T1, T2, [1]) ≈ T2."""
    xp = np
    dtype = np.dtype(f"float{precision}")

    se3_1 = random_se3(batch_dims=batch_dims, precision=precision)
    se3_2 = random_se3(batch_dims=batch_dims, precision=precision)

    t_0 = xp.asarray([0.0], dtype=dtype)
    t_1 = xp.asarray([1.0], dtype=dtype)

    result_0 = SE3.slerp(se3_1, se3_2, t_0)
    result_1 = SE3.slerp(se3_1, se3_2, t_1)

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


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_slerp_midpoint_translation(batch_dims, precision):
    """Test that the midpoint translation is the average of endpoints."""
    xp = np
    dtype = np.dtype(f"float{precision}")

    se3_1 = random_se3(batch_dims=batch_dims, precision=precision)
    se3_2 = random_se3(batch_dims=batch_dims, precision=precision)

    t_half = xp.asarray([0.5], dtype=dtype)
    result = SE3.slerp(se3_1, se3_2, t_half)

    se3_1_np = np.array(se3_1)
    se3_2_np = np.array(se3_2)
    result_np = np.array(result).squeeze(-2)

    expected_t = (se3_1_np[..., 4:7] + se3_2_np[..., 4:7]) / 2.0
    assert np.allclose(result_np[..., 4:7], expected_t, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_slerp_same_transform(batch_dims, precision):
    """Slerping between the same transform should return that transform."""
    xp = np
    dtype = np.dtype(f"float{precision}")

    se3 = random_se3(batch_dims=batch_dims, precision=precision)
    t_vals = xp.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=dtype)

    result = SE3.slerp(se3, se3, t_vals)

    assert result.shape == batch_dims + (5, 7)

    se3_np = np.array(se3)
    result_np = np.array(result)

    for i in range(5):
        r = result_np[..., i, :]
        dot = np.sum(se3_np[..., :4] * r[..., :4], axis=-1)
        assert np.allclose(np.abs(dot), 1.0, atol=ATOL[precision])
        assert np.allclose(se3_np[..., 4:7], r[..., 4:7], atol=ATOL[precision])
