"""Tests for SO3.random uniform sampling."""

import numpy as np
import pytest
from conftest import ATOL, TEST_BATCH_DIMS, TEST_PRECISIONS

from nanomanifold import SO3


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_random_is_canonical_unit_quaternion(batch_dims, precision):
    q = SO3.random(*batch_dims, dtype=np.dtype(f"float{precision}"))

    assert q.shape == batch_dims + (4,)
    assert q.dtype == np.dtype(f"float{precision}")
    assert np.allclose(np.linalg.norm(q, axis=-1), 1.0, atol=ATOL[precision])
    assert np.all(q[..., 0] >= 0)


def test_random_statistical_uniformity():
    """The rotation angle of a uniform rotation has density (1 - cos θ) / π on [0, π]."""
    angles = np.linalg.norm(SO3.to_axis_angle(SO3.random(20000)), axis=-1)
    # E[θ] = π / 2 + 2 / π for the uniform distribution on SO(3).
    assert abs(angles.mean() - (np.pi / 2 + 2 / np.pi)) < 0.02
