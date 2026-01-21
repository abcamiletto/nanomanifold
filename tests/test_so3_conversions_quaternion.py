import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PRECISIONS, random_quaternion

from nanomanifold import SO3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_quaternion_xyzw_roundtrip(backend, batch_dims, precision):
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    quat_xyzw = SO3.to_quat_xyzw(quat)
    assert quat_xyzw.dtype == quat.dtype
    assert quat_xyzw.shape == quat.shape

    quat_converted = SO3.from_quat_xyzw(quat_xyzw)
    assert quat_converted.dtype == quat.dtype
    assert quat_converted.shape == quat.shape

    quat_np = np.array(quat)
    quat_converted_np = np.array(quat_converted)
    assert np.allclose(quat_np, quat_converted_np, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_quaternion_xyzw_canonicalizes(backend, batch_dims, precision):
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)
    quat_neg = -quat

    quat_xyzw = SO3.to_quat_xyzw(quat_neg)
    quat_xyzw_np = np.array(quat_xyzw)
    assert np.all(quat_xyzw_np[..., 3] >= -ATOL[precision])
