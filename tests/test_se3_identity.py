import numpy as np
import pytest
from conftest import TEST_BATCH_DIMS, TEST_PRECISIONS, identity_se3, random_points, random_se3

from nanomanifold import SE3


@pytest.mark.parametrize("convention,expected", [("wxyz", [1, 0, 0, 0, 0, 0, 0]), ("xyzw", [0, 0, 0, 1, 0, 0, 0])])
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_identity_as(convention, expected, batch_dims, precision):
    ref = random_se3(batch_dims=batch_dims, precision=precision)

    identity = SE3.identity_as(ref, batch_dims=batch_dims, convention=convention)

    assert identity.dtype == ref.dtype
    assert identity.shape == batch_dims + (7,)
    assert np.array_equal(identity, np.broadcast_to(np.array(expected, dtype=ref.dtype), batch_dims + (7,)))


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_identity_as_is_neutral(batch_dims):
    T = random_se3(batch_dims=batch_dims, precision=64)
    identity = SE3.identity_as(T, batch_dims=batch_dims)
    points = random_points(batch_dims=batch_dims, precision=64)

    assert np.allclose(identity, identity_se3(batch_dims, precision=64))
    assert np.allclose(SE3.to_matrix(SE3.multiply(T, identity)), SE3.to_matrix(T))
    assert np.allclose(SE3.to_matrix(SE3.multiply(identity, T)), SE3.to_matrix(T))
    assert np.allclose(SE3.transform_points(identity, points), points)


def test_identity_as_uses_explicit_batch_dims():
    identity = SE3.identity_as(np.zeros((4, 4), dtype=np.float32), batch_dims=(2, 5))

    assert identity.shape == (2, 5, 7)
    assert identity.dtype == np.float32
