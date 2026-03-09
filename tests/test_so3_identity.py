import numpy as np
import pytest
from conftest import TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PASS_XP, TEST_PRECISIONS, get_xp_kwargs, random_quaternion

from nanomanifold import SO3


@pytest.mark.parametrize(
    "rotation_type,expected_tail,expected",
    [
        ("quat", (4,), [1.0, 0.0, 0.0, 0.0]),
        ("axis_angle", (3,), [0.0, 0.0, 0.0]),
        ("euler", (3,), [0.0, 0.0, 0.0]),
        ("sixd", (6,), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    ],
)
@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_identity_as_vector_representations(rotation_type, expected_tail, expected, backend, batch_dims, precision, pass_xp):
    ref = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)
    xp_kwargs = get_xp_kwargs(backend, pass_xp)

    identity = SO3.identity_as(ref, batch_dims=batch_dims, rotation_type=rotation_type, **xp_kwargs)

    assert identity.dtype == ref.dtype
    assert identity.shape == batch_dims + expected_tail
    assert np.allclose(np.array(identity), np.array(expected, dtype=np.array(identity).dtype))


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_identity_as_matrix(backend, batch_dims, precision, pass_xp):
    ref = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)
    xp_kwargs = get_xp_kwargs(backend, pass_xp)

    identity = SO3.identity_as(ref, batch_dims=batch_dims, rotation_type="matrix", **xp_kwargs)

    assert identity.dtype == ref.dtype
    assert identity.shape == batch_dims + (3, 3)
    assert np.allclose(np.array(identity), np.broadcast_to(np.eye(3, dtype=np.array(identity).dtype), batch_dims + (3, 3)))


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_identity_as_uses_explicit_batch_dims(backend, batch_dims, precision, pass_xp):
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)
    ref = SO3.to_matrix(quat, **get_xp_kwargs(backend, pass_xp))
    xp_kwargs = get_xp_kwargs(backend, pass_xp)

    identity = SO3.identity_as(ref, batch_dims=batch_dims, **xp_kwargs)

    assert identity.dtype == ref.dtype
    assert identity.shape == batch_dims + (4,)
    assert np.allclose(np.array(identity), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.array(identity).dtype))
