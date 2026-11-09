import numpy as np
import pytest
from conftest import ATOL, TEST_BATCH_DIMS, TEST_PRECISIONS, random_quaternion
from scipy.spatial.transform import Rotation as R

from nanomanifold import SO3


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_inverse_identity(batch_dims, precision):
    # Create a random SO3 quaternion
    quat = random_quaternion(batch_dims=batch_dims, precision=precision)

    # Compute inverse
    quat_inv = SO3.inverse(quat)

    assert quat_inv.dtype == quat.dtype
    assert quat_inv.shape == quat.shape

    # Test that q * q_inv = identity quaternion [1, 0, 0, 0]
    # For quaternions, multiplication is not commutative, but for rotations q * q^-1 = identity
    # We can verify by converting to rotation matrices and checking R * R_inv = I
    rotmat_orig = SO3.to_rotmat(quat)
    rotmat_inv = SO3.to_rotmat(quat_inv)

    # Compute product R * R_inv (should be identity)
    identity = np.matmul(np.array(rotmat_orig), np.array(rotmat_inv))

    # Check if result is identity matrix
    expected_identity = np.eye(3)
    for _ in range(len(batch_dims)):
        expected_identity = np.expand_dims(expected_identity, axis=0)
    expected_identity = np.broadcast_to(expected_identity, identity.shape)

    assert np.allclose(identity, expected_identity, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_inverse_scipy(batch_dims):
    # Create a random SO3 quaternion
    quat = random_quaternion(batch_dims=batch_dims, precision=32)

    # Compute inverse using nanomanifold
    quat_inv = SO3.inverse(quat)

    # Compute inverse using scipy
    quat_np = np.array(quat)
    # Convert from [w, x, y, z] to scipy's [x, y, z, w] format
    quat_scipy = np.concatenate([quat_np[..., 1:4], quat_np[..., 0:1]], axis=-1)
    r = R.from_quat(quat_scipy.reshape(-1, 4))
    r_inv = r.inv()
    quat_inv_scipy = r_inv.as_quat().reshape(quat_scipy.shape)
    # Convert back to [w, x, y, z] format
    quat_inv_scipy = np.concatenate([quat_inv_scipy[..., 3:4], quat_inv_scipy[..., 0:3]], axis=-1)

    assert quat_inv.dtype == quat.dtype
    assert quat_inv.shape == quat_inv_scipy.shape

    quat_inv_np = np.array(quat_inv)
    # Check quaternion equivalence (q and -q represent the same rotation)
    dot_products = np.sum(quat_inv_np * quat_inv_scipy, axis=-1)
    assert np.allclose(np.abs(dot_products), 1.0, atol=1e-6)
