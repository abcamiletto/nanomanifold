"""SO3.convert: every edge of the representation tree, and every route through it."""

import numpy as np
import pytest
from conftest import ATOL, TEST_PRECISIONS, awkward_quaternions, random_quaternion, without_singularities
from scipy.spatial.transform import Rotation as R

from nanomanifold import SO3
from nanomanifold.SO3.primitives.euler import EULER_CONVENTIONS

REPS = ["axis_angle", "euler", "hinge", "matrix", "rotmat", "quat", "sixd"]
AXES = np.array([0.0, 0.0, 2.0])

# (representation, options) pairs that round-trip any rotation through the canonical quaternion.
ROUND_TRIPS = [
    ("quat", {"convention": "wxyz"}),
    ("quat", {"convention": "xyzw"}),
    ("axis_angle", {}),
    ("rotmat", {}),
    ("sixd", {}),
    *[("euler", {"convention": convention}) for convention in EULER_CONVENTIONS],
]


def _options(rep, dtype):
    return {"axes": AXES.astype(dtype)} if rep == "hinge" else {}


def _as_rotmat(x, rep, options):
    return SO3.convert(x, src=rep, dst="rotmat", src_kwargs=options)


def _same_rotation(q1, q2, atol):
    return np.allclose(np.abs(np.sum(q1 * q2, axis=-1)), 1.0, atol=atol)


@pytest.mark.parametrize("rep,options", ROUND_TRIPS, ids=[f"{rep}-{options}" for rep, options in ROUND_TRIPS])
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_round_trip(rep, options, precision):
    quat = without_singularities(awkward_quaternions(precision), rep, options)

    x = SO3.convert(quat, src="quat", dst=rep, dst_kwargs=options)
    back = SO3.convert(x, src=rep, dst="quat", src_kwargs=options)

    assert x.dtype == quat.dtype
    assert back.dtype == quat.dtype
    assert back.shape == quat.shape
    assert _same_rotation(back, quat, ATOL[precision])


@pytest.mark.parametrize("src,dst", [(src, dst) for src in REPS for dst in REPS], ids=lambda rep: rep)
def test_route_matches_quaternion_detour(src, dst):
    """Each pair has its own route through the tree; all of them must agree on the rotation."""
    quat = without_singularities(without_singularities(awkward_quaternions(), src, {}), dst, {})
    src_options = _options(src, quat.dtype)
    dst_options = _options(dst, quat.dtype)
    x = SO3.convert(quat, src="quat", dst=src, dst_kwargs=src_options)

    direct = SO3.convert(x, src=src, dst=dst, src_kwargs=src_options, dst_kwargs=dst_options)
    via_quat = SO3.convert(SO3.convert(x, src=src, dst="quat", src_kwargs=src_options), src="quat", dst=dst, dst_kwargs=dst_options)

    assert direct.dtype == x.dtype
    assert direct.shape == via_quat.shape
    assert np.allclose(_as_rotmat(direct, dst, dst_options), _as_rotmat(via_quat, dst, dst_options), atol=1e-9)


@pytest.mark.parametrize(
    "rep,options,scipy_rep",
    [
        ("axis_angle", {}, lambda r: r.as_rotvec()),
        ("rotmat", {}, lambda r: r.as_matrix()),
        ("quat", {"convention": "xyzw"}, lambda r: r.as_quat(canonical=False)),
        *[("euler", {"convention": c}, lambda r, c=c: r.as_euler(c)) for c in EULER_CONVENTIONS],
    ],
    ids=lambda p: p if isinstance(p, str) else "",
)
def test_matches_scipy(rep, options, scipy_rep):
    quat = random_quaternion(batch_dims=(16,), precision=64)

    result = SO3.convert(quat, src="quat", dst=rep, dst_kwargs=options)
    rotation = R.from_quat(np.concatenate([quat[..., 1:], quat[..., :1]], axis=-1))

    assert np.allclose(result, scipy_rep(rotation), atol=1e-9)


def test_sixd_is_first_two_rotmat_columns():
    rotmat = SO3.to_rotmat(awkward_quaternions())

    sixd = SO3.convert(rotmat, src="rotmat", dst="sixd")

    assert np.array_equal(sixd, np.concatenate([rotmat[..., :, 0], rotmat[..., :, 1]], axis=-1))


def test_sixd_to_rotmat_is_gram_schmidt():
    sixd = np.random.default_rng(0).normal(size=(64, 6))

    rotmat = SO3.convert(sixd, src="sixd", dst="matrix")

    b1 = sixd[:, :3] / np.linalg.norm(sixd[:, :3], axis=-1, keepdims=True)
    b2 = sixd[:, 3:] - np.sum(b1 * sixd[:, 3:], axis=-1, keepdims=True) * b1
    b2 /= np.linalg.norm(b2, axis=-1, keepdims=True)
    assert np.allclose(rotmat, np.stack([b1, b2, np.cross(b1, b2)], axis=-1), atol=1e-12)


@pytest.mark.parametrize("mode", ["svd", "davenport"])
def test_matrix_projects_to_nearest_rotation(mode):
    rotmat = SO3.to_rotmat(awkward_quaternions())
    stretched = rotmat @ np.diag([1.05, 0.97, 1.02])

    projected = SO3.convert(stretched, src="matrix", dst="rotmat", src_kwargs={"mode": mode})

    assert np.allclose(projected, rotmat, atol=1e-9)


def test_davenport_matches_svd_for_orientation_reversing_matrix():
    matrix = np.array([[1.0, 0.2, 0.1], [0.3, 2.0, -0.4], [0.2, 0.1, -1.0]])

    davenport = SO3.convert(matrix, src="matrix", dst="rotmat", src_kwargs={"mode": "davenport"})
    svd = SO3.convert(matrix, src="matrix", dst="rotmat", src_kwargs={"mode": "svd"})

    assert np.allclose(davenport, svd, atol=ATOL[64])


def test_matrix_target_is_rotmat():
    quat = awkward_quaternions()

    assert np.array_equal(SO3.convert(quat, src="quat", dst="matrix"), SO3.convert(quat, src="quat", dst="rotmat"))


def test_same_representation_applies_option_changes():
    quat = awkward_quaternions()
    euler_xyz = SO3.to_euler(quat, convention="XYZ")
    quat_xyzw = SO3.to_quat(-quat, convention="xyzw")

    euler_zyx = SO3.convert(euler_xyz, src="euler", dst="euler", src_kwargs={"convention": "XYZ"}, dst_kwargs={"convention": "ZYX"})
    quat_wxyz = SO3.convert(quat_xyzw, src="quat", dst="quat", src_kwargs={"convention": "xyzw"})

    assert _same_rotation(SO3.from_euler(euler_zyx, convention="ZYX"), quat, 1e-9)
    assert np.allclose(quat_wxyz, quat, atol=1e-12)


def test_quaternions_are_canonicalized():
    quat = -awkward_quaternions()

    xyzw = SO3.to_quat(quat, convention="xyzw")

    assert np.all(xyzw[..., 3] >= 0)
    assert np.allclose(np.linalg.norm(xyzw, axis=-1), 1.0)


def test_hinge_to_axis_angle_is_unwrapped_scaling():
    angles = np.array([[0.1], [-0.2], [4.0]])

    axis_angle = SO3.convert(angles, src="hinge", dst="axis_angle", src_kwargs={"axes": AXES})

    assert np.array_equal(axis_angle, angles * AXES)


def test_hinge_is_least_squares_coefficient_along_axes():
    axis_angle = np.array([[0.2, 0.3, 0.0], [-0.4, 0.1, 0.0]])
    axes = np.array([2.0, 0.0, 0.0])

    angles = SO3.convert(axis_angle, src="axis_angle", dst="hinge", dst_kwargs={"axes": axes})

    assert np.allclose(angles, [[0.1], [-0.2]])


@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_hinge_round_trip_with_batched_nonunit_axes(precision):
    angles = np.array([[0.1], [-0.2], [0.3]], dtype=f"float{precision}")
    axes = np.array([[0.0, 0.0, 2.0], [0.0, 1.0, 0.0], [3.0, 0.0, 4.0]], dtype=f"float{precision}") / 5

    quat = SO3.from_hinge(angles, axes, convention="xyzw")
    recovered = SO3.to_hinge(quat, axes, convention="xyzw")

    assert recovered.dtype == angles.dtype
    assert np.allclose(recovered, angles, atol=ATOL[precision])


@pytest.mark.filterwarnings("ignore:Gimbal lock detected")
@pytest.mark.parametrize("convention", EULER_CONVENTIONS)
def test_euler_at_gimbal_lock_matches_scipy(convention):
    """Only the sum or difference of the outer angles is defined; like SciPy, the third angle is set to zero."""
    proper = convention[0].lower() == convention[2].lower()
    locks = [0.0, np.pi] if proper else [np.pi / 2, -np.pi / 2]
    euler = np.array([[first, middle, third] for middle in locks for first, third in [(0.3, -0.7), (0.0, 0.0), (-2.0, 1.1)]])
    quat = SO3.from_euler(euler, convention=convention)

    result = SO3.to_euler(quat, convention=convention)

    expected = R.from_quat(np.concatenate([quat[..., 1:], quat[..., :1]], axis=-1)).as_euler(convention)
    assert np.allclose(result, expected, atol=1e-9)
    assert _same_rotation(SO3.from_euler(result, convention=convention), quat, 1e-12)


def test_float16_svd_projection_is_rejected():
    with pytest.raises(ValueError, match="does not support float16"):
        SO3.convert(np.eye(3, dtype=np.float16), src="matrix", dst="rotmat")


@pytest.mark.parametrize(
    "src,options,error",
    [
        ("hinge", {}, TypeError),
        ("matrix", {"mode": "bad"}, ValueError),
        ("euler", {"convention": "bad"}, AssertionError),
        ("quat", {"convention": "bad"}, AssertionError),
    ],
)
def test_invalid_options_raise(src, options, error):
    with pytest.raises(error):
        SO3.convert(np.zeros((3, 3) if src == "matrix" else (4,)), src=src, dst="rotmat", src_kwargs=options)
