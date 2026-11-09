import numpy as np
import pytest
from conftest import (
    ATOL,
    TEST_BATCH_DIMS,
    TEST_PRECISIONS,
    identity_quaternion,
    random_quaternion,
)
from scipy.spatial.transform import Rotation as R

from nanomanifold import SO3


@pytest.mark.parametrize(
    "rotation_type,convention",
    [
        ("quat", "wxyz"),
        ("quat", "xyzw"),
        ("axis_angle", "wxyz"),
        ("euler", "ZYX"),
        ("hinge", "wxyz"),
        ("rotmat", "wxyz"),
        ("matrix", "wxyz"),
        ("sixd", "wxyz"),
    ],
)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_distance_rotation_type_matches_quaternion_reference(rotation_type, convention, batch_dims):
    q1 = random_quaternion(batch_dims=batch_dims, precision=32)
    q2 = random_quaternion(batch_dims=batch_dims, precision=32)

    rot_kwargs = {"convention": convention} if rotation_type in ("quat", "euler") else {}
    if rotation_type == "quat":
        x1 = SO3.to_quat(q1, convention=convention)
        x2 = SO3.to_quat(q2, convention=convention)
    elif rotation_type == "axis_angle":
        x1 = SO3.to_axis_angle(q1)
        x2 = SO3.to_axis_angle(q2)
    elif rotation_type == "euler":
        x1 = SO3.to_euler(q1, convention=convention)
        x2 = SO3.to_euler(q2, convention=convention)
    elif rotation_type == "hinge":
        xp = np
        axes = xp.asarray(np.array([0.0, 0.0, 2.0], dtype=np.float32))
        x1 = SO3.to_hinge(q1, axes)
        x2 = SO3.to_hinge(q2, axes)
        rot_kwargs = {"axes": axes}
    elif rotation_type == "rotmat":
        x1 = SO3.to_rotmat(q1)
        x2 = SO3.to_rotmat(q2)
    elif rotation_type == "matrix":
        xp = np
        stretch = np.diag(np.array([1.05, 0.97, 1.02], dtype=np.float32))
        x1 = xp.asarray(np.matmul(np.array(SO3.to_rotmat(q1)), stretch))
        x2 = xp.asarray(np.matmul(np.array(SO3.to_rotmat(q2)), stretch))
    else:
        x1 = SO3.to_sixd(q1)
        x2 = SO3.to_sixd(q2)

    distance = SO3.distance(x1, x2, rotation_type=rotation_type, rot_kwargs=rot_kwargs)
    if rotation_type == "hinge":
        expected = SO3.distance(SO3.from_hinge(x1, axes), SO3.from_hinge(x2, axes))
    else:
        expected = SO3.distance(q1, q2)

    assert distance.shape == expected.shape
    assert np.allclose(np.array(distance), np.array(expected), atol=ATOL[32])


def test_distance_supports_euler_convention():
    q1 = random_quaternion(batch_dims=(8,), precision=32)
    q2 = random_quaternion(batch_dims=(8,), precision=32)

    e1 = SO3.to_euler(q1, convention="XYZ")
    e2 = SO3.to_euler(q2, convention="XYZ")

    distance = SO3.distance(e1, e2, rotation_type="euler", rot_kwargs={"convention": "XYZ"})
    expected = SO3.distance(q1, q2)

    assert np.allclose(np.array(distance), np.array(expected), atol=ATOL[32])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_distance_identity(batch_dims, precision):
    # Distance from any quaternion to itself should be 0
    quat = random_quaternion(batch_dims=batch_dims, precision=precision)
    distance = SO3.distance(quat, quat)

    assert distance.dtype == quat.dtype
    assert distance.shape == batch_dims

    distance_np = np.array(distance)

    # Allow appropriate tolerance for numerical precision
    assert np.allclose(distance_np, 0.0, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_distance_identity_quaternion(batch_dims, precision):
    # Distance between any quaternion and identity should equal distance to canonical identity
    quat = random_quaternion(batch_dims=batch_dims, precision=precision)
    identity = identity_quaternion(batch_dims=batch_dims, precision=precision)

    distance = SO3.distance(quat, identity)

    assert distance.dtype == quat.dtype
    assert distance.shape == batch_dims

    distance_np = np.array(distance)

    # Distance should be in [0, π]
    tol = ATOL[precision]
    assert np.all(distance_np >= -tol)
    assert np.all(distance_np <= np.pi + tol)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_distance_symmetry(batch_dims, precision):
    # Distance should be symmetric: d(q1, q2) = d(q2, q1)
    q1 = random_quaternion(batch_dims=batch_dims, precision=precision)
    q2 = random_quaternion(batch_dims=batch_dims, precision=precision)

    dist1 = SO3.distance(q1, q2)
    dist2 = SO3.distance(q2, q1)

    assert dist1.dtype == q1.dtype
    assert dist1.shape == batch_dims
    assert dist2.dtype == q1.dtype
    assert dist2.shape == batch_dims

    dist1_np = np.array(dist1)
    dist2_np = np.array(dist2)
    assert np.allclose(dist1_np, dist2_np, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_distance_quaternion_double_cover(batch_dims, precision):
    # Distance between q and -q should be 0 (same rotation)
    quat = random_quaternion(batch_dims=batch_dims, precision=precision)
    quat_neg = -quat

    distance = SO3.distance(quat, quat_neg)

    assert distance.dtype == quat.dtype
    assert distance.shape == batch_dims

    distance_np = np.array(distance)
    # Allow appropriate tolerance for numerical precision
    assert np.allclose(distance_np, 0.0, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_distance_scipy(batch_dims):
    # Compare with scipy's angle calculation
    q1 = random_quaternion(batch_dims=batch_dims, precision=32)
    q2 = random_quaternion(batch_dims=batch_dims, precision=32)

    # Compute distance using nanomanifold
    distance = SO3.distance(q1, q2)

    # Convert to numpy for scipy
    q1_np = np.array(q1)
    q2_np = np.array(q2)

    # Convert from [w, x, y, z] to scipy's [x, y, z, w] format
    q1_scipy = np.concatenate([q1_np[..., 1:4], q1_np[..., 0:1]], axis=-1)
    q2_scipy = np.concatenate([q2_np[..., 1:4], q2_np[..., 0:1]], axis=-1)

    # Reshape for scipy processing
    q1_flat = q1_scipy.reshape(-1, 4)
    q2_flat = q2_scipy.reshape(-1, 4)

    # Compute angles using scipy
    scipy_angles = []
    for i in range(len(q1_flat)):
        r1 = R.from_quat(q1_flat[i : i + 1])
        r2 = R.from_quat(q2_flat[i : i + 1])
        # Compute relative rotation
        r_rel = r1.inv() * r2
        # Get magnitude of rotation angle
        angle = np.abs(r_rel.magnitude())
        scipy_angles.append(angle)

    scipy_angles = np.array(scipy_angles).reshape(batch_dims)

    assert distance.dtype == q1.dtype
    assert distance.shape == scipy_angles.shape

    distance_np = np.array(distance)
    assert np.allclose(distance_np, scipy_angles, atol=1e-6)


def test_distance_specific_rotations():
    # Test distance between specific known rotations
    identity = np.array([1.0, 0.0, 0.0, 0.0])

    # 90-degree rotation around z-axis: angle = π/2
    sqrt2_inv = 1.0 / np.sqrt(2.0)
    q_z90 = np.array([sqrt2_inv, 0, 0, sqrt2_inv])

    # 180-degree rotation around z-axis: angle = π
    q_z180 = np.array([0.0, 0.0, 0.0, 1.0])

    # Distance from identity to 90° rotation should be π/2
    dist_90 = SO3.distance(identity, q_z90)
    dist_90_np = np.array(dist_90)
    assert np.allclose(dist_90_np, np.pi / 2, atol=1e-6)

    # Distance from identity to 180° rotation should be π
    dist_180 = SO3.distance(identity, q_z180)
    dist_180_np = np.array(dist_180)
    assert np.allclose(dist_180_np, np.pi, atol=1e-6)

    # Distance from 90° to 180° should be π/2
    dist_90_to_180 = SO3.distance(q_z90, q_z180)
    dist_90_to_180_np = np.array(dist_90_to_180)
    assert np.allclose(dist_90_to_180_np, np.pi / 2, atol=1e-6)


def test_distance_range():
    # Distance should always be in range [0, π]
    q1 = random_quaternion(batch_dims=(100,), precision=32)
    q2 = random_quaternion(batch_dims=(100,), precision=32)

    distances = SO3.distance(q1, q2)
    distances_np = np.array(distances)

    assert np.all(distances_np >= 0.0)
    assert np.all(distances_np <= np.pi + 1e-6)


def test_distance_triangle_inequality():
    # Triangle inequality: d(q1,q3) <= d(q1,q2) + d(q2,q3)
    q1 = random_quaternion(batch_dims=(10,), precision=32)
    q2 = random_quaternion(batch_dims=(10,), precision=32)
    q3 = random_quaternion(batch_dims=(10,), precision=32)

    d12 = np.array(SO3.distance(q1, q2))
    d23 = np.array(SO3.distance(q2, q3))
    d13 = np.array(SO3.distance(q1, q3))

    # Triangle inequality should hold (with small tolerance for numerical errors)
    assert np.all(d13 <= d12 + d23 + 1e-6)


def test_distance_broadcasting():
    # Test broadcasting between different batch sizes
    q_single = random_quaternion(batch_dims=(), precision=32)
    q_batch = random_quaternion(batch_dims=(3,), precision=32)

    # Test broadcasting
    distance = SO3.distance(q_single, q_batch)

    assert distance.shape == (3,)

    distance_np = np.array(distance)
    assert np.all(distance_np >= 0.0)
    assert np.all(distance_np <= np.pi + 1e-6)
