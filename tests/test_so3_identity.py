import numpy as np
import pytest
from conftest import TEST_BATCH_DIMS, TEST_PRECISIONS, random_quaternion

from nanomanifold import SO3


@pytest.mark.parametrize(
    "rep,rep_kwargs,expected_tail,expected",
    [
        ("quat", {}, (4,), [1.0, 0.0, 0.0, 0.0]),
        ("quat", {"convention": "xyzw"}, (4,), [0.0, 0.0, 0.0, 1.0]),
        ("axis_angle", {}, (3,), [0.0, 0.0, 0.0]),
        ("hinge", {}, (1,), [0.0]),
        ("hinge", {"axes": np.array([0.0, 0.0, 2.0])}, (1,), [0.0]),
        ("euler", {}, (3,), [0.0, 0.0, 0.0]),
        ("euler", {"convention": "ZXZ"}, (3,), [0.0, 0.0, 0.0]),
        ("sixd", {}, (6,), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    ],
)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_identity_as_vector_representations(rep, rep_kwargs, expected_tail, expected, batch_dims, precision):
    ref = random_quaternion(batch_dims=batch_dims, precision=precision)

    identity = SO3.identity_as(ref, batch_dims=batch_dims, rep=rep, rep_kwargs=rep_kwargs)

    assert identity.dtype == ref.dtype
    assert identity.shape == batch_dims + expected_tail
    assert np.allclose(np.array(identity), np.array(expected, dtype=np.array(identity).dtype))


@pytest.mark.parametrize("rep,rep_kwargs", [("matrix", {}), ("matrix", {"mode": "davenport"}), ("rotmat", {})])
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_identity_as_matrix_representations(rep, rep_kwargs, batch_dims, precision):
    ref = random_quaternion(batch_dims=batch_dims, precision=precision)

    identity = SO3.identity_as(ref, batch_dims=batch_dims, rep=rep, rep_kwargs=rep_kwargs)

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


@pytest.mark.parametrize(
    "rep,rep_kwargs,error",
    [
        ("quat", {"convention": "bad"}, AssertionError),
        ("euler", {"convention": "bad"}, AssertionError),
        ("matrix", {"mode": "bad"}, ValueError),
        ("bad", {}, ValueError),
    ],
)
def test_identity_as_rejects_bad_options(rep, rep_kwargs, error):
    with pytest.raises(error):
        SO3.identity_as(random_quaternion(), batch_dims=(), rep=rep, rep_kwargs=rep_kwargs)
