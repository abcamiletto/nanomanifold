import numpy as np
import pytest
from conftest import TEST_BATCH_DIMS, TEST_PRECISIONS, random_quaternion

from nanomanifold import SO3


@pytest.mark.parametrize(
    "rotation_type,convention,expected_tail,expected",
    [
        ("quat", "wxyz", (4,), [1.0, 0.0, 0.0, 0.0]),
        ("quat", "xyzw", (4,), [0.0, 0.0, 0.0, 1.0]),
        ("axis_angle", "wxyz", (3,), [0.0, 0.0, 0.0]),
        ("hinge", "wxyz", (1,), [0.0]),
        ("euler", "ZYX", (3,), [0.0, 0.0, 0.0]),
        ("sixd", "wxyz", (6,), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    ],
)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_identity_as_vector_representations(rotation_type, convention, expected_tail, expected, batch_dims, precision):
    ref = random_quaternion(batch_dims=batch_dims, precision=precision)

    identity = SO3.identity_as(ref, batch_dims=batch_dims, rotation_type=rotation_type, convention=convention)

    assert identity.dtype == ref.dtype
    assert identity.shape == batch_dims + expected_tail
    assert np.allclose(np.array(identity), np.array(expected, dtype=np.array(identity).dtype))


@pytest.mark.parametrize("rotation_type", ["matrix", "rotmat"])
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_identity_as_matrix_representations(rotation_type, batch_dims, precision):
    ref = random_quaternion(batch_dims=batch_dims, precision=precision)

    identity = SO3.identity_as(ref, batch_dims=batch_dims, rotation_type=rotation_type)

    assert identity.dtype == ref.dtype
    assert identity.shape == batch_dims + (3, 3)
    assert np.allclose(np.array(identity), np.broadcast_to(np.eye(3, dtype=np.array(identity).dtype), batch_dims + (3, 3)))


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_identity_as_uses_explicit_batch_dims(batch_dims, precision):
    quat = random_quaternion(batch_dims=batch_dims, precision=precision)
    ref = SO3.to_rotmat(quat)

    identity = SO3.identity_as(ref, batch_dims=batch_dims)

    assert identity.dtype == ref.dtype
    assert identity.shape == batch_dims + (4,)
    assert np.allclose(np.array(identity), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.array(identity).dtype))
