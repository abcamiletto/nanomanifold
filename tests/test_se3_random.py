"""Tests for SE3.random sampling."""

import numpy as np
import pytest
from conftest import ATOL, TEST_BATCH_DIMS, TEST_PRECISIONS

from nanomanifold import SE3


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_random_rotation_and_translation(batch_dims, precision):
    se3 = SE3.random(*batch_dims, dtype=np.dtype(f"float{precision}"))
    quat = se3[..., :4]
    translation = se3[..., 4:]

    assert se3.shape == batch_dims + (7,)
    assert se3.dtype == np.dtype(f"float{precision}")
    assert np.allclose(np.linalg.norm(quat, axis=-1), 1.0, atol=ATOL[precision])
    assert np.all(quat[..., 0] >= 0)
    assert np.all((translation >= -1) & (translation <= 1))
