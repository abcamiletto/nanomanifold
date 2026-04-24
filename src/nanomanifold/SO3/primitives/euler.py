from types import ModuleType
from typing import Any, Literal

from jaxtyping import Float

from nanomanifold import common
from nanomanifold.common import get_namespace

from ..identity import identity_as
from ..multiply import multiply
from . import rotmat
from .quaternion import QuaternionConvention, canonicalize, from_quat, to_quat

EulerConvention = Literal[
    "xyz",
    "xzy",
    "yxz",
    "yzx",
    "zxy",
    "zyx",
    "XYZ",
    "XZY",
    "YXZ",
    "YZX",
    "ZXY",
    "ZYX",
    "xyx",
    "xzx",
    "yxy",
    "yzy",
    "zxz",
    "zyz",
    "XYX",
    "XZX",
    "YXY",
    "YZY",
    "ZXZ",
    "ZYZ",
]

_EULER_CONVENTIONS = (
    "xyz",
    "xzy",
    "yxz",
    "yzx",
    "zxy",
    "zyx",
    "XYZ",
    "XZY",
    "YXZ",
    "YZX",
    "ZXY",
    "ZYX",
    "xyx",
    "xzx",
    "yxy",
    "yzy",
    "zxz",
    "zyz",
    "XYX",
    "XZX",
    "YXY",
    "YZY",
    "ZXZ",
    "ZYZ",
)


def to_euler(
    q: Float[Any, "... 4"],
    *,
    convention: EulerConvention = "ZYX",
    quat_convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 3"]:
    assert convention in _EULER_CONVENTIONS, "Invalid Euler convention."
    assert quat_convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace(q)
    q = canonicalize(from_quat(q, convention=quat_convention, xp=xp), xp=xp)
    rot = rotmat.to_rotmat(q, xp=xp)
    return _rotmat_to_euler(rot, convention, xp=xp)


def from_euler(
    euler: Float[Any, "... 3"],
    *,
    convention: EulerConvention = "ZYX",
    quat_convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    assert convention in _EULER_CONVENTIONS, "Invalid Euler convention."
    assert quat_convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace(euler)
    half_angles = euler * 0.5
    cos_half = xp.cos(half_angles)
    sin_half = xp.sin(half_angles)

    ones = xp.ones_like(euler[..., :1])
    zeros = xp.zeros_like(euler)
    q = xp.concatenate([ones, zeros], axis=-1)

    is_extrinsic = convention.islower()
    conv = convention.lower()

    for i, axis in enumerate(conv):
        q_axis = _axis_quaternion(cos_half[..., i], sin_half[..., i], axis, xp)
        q = multiply(q_axis, q, xp=xp) if is_extrinsic else multiply(q, q_axis, xp=xp)

    return to_quat(canonicalize(q, xp=xp), convention=quat_convention, xp=xp)


def _axis_quaternion(cos_half: Float[Any, "..."], sin_half: Float[Any, "..."], axis: str, xp: ModuleType) -> Float[Any, "... 4"]:
    zero = xp.zeros_like(cos_half)
    if axis == "x":
        return xp.stack([cos_half, sin_half, zero, zero], axis=-1)
    if axis == "y":
        return xp.stack([cos_half, zero, sin_half, zero], axis=-1)
    if axis == "z":
        return xp.stack([cos_half, zero, zero, sin_half], axis=-1)
    raise ValueError(f"Invalid axis: {axis}")


def _euler_to_rotmat(euler: Float[Any, "... 3"], convention: EulerConvention, *, xp: ModuleType | None = None) -> Float[Any, "... 3 3"]:
    """Convert Euler angles to a rotation matrix."""
    assert convention in _EULER_CONVENTIONS, "Invalid Euler convention."
    if xp is None:
        xp = get_namespace(euler)

    rot = identity_as(euler, batch_dims=euler.shape[:-1], rotation_type="rotmat", xp=xp)

    is_extrinsic = convention.islower()
    conv = convention.lower()

    for i, axis in enumerate(conv):
        angle = euler[..., i]
        R_axis = _rotation_matrix(angle, axis, xp=xp)

        if is_extrinsic:
            rot = xp.matmul(R_axis, rot)
        else:
            rot = xp.matmul(rot, R_axis)

    return rot


def _rotation_matrix(angle: Float[Any, "..."], axis: str, *, xp: ModuleType | None = None) -> Float[Any, "... 3 3"]:
    """Create rotation matrix for given angle and axis."""
    if xp is None:
        xp = get_namespace(angle)

    cos_a = xp.cos(angle)
    sin_a = xp.sin(angle)
    zero = xp.zeros_like(cos_a)
    ones = xp.ones_like(cos_a)

    if axis == "x":
        mat = xp.stack(
            [xp.stack([ones, zero, zero], axis=-1), xp.stack([zero, cos_a, -sin_a], axis=-1), xp.stack([zero, sin_a, cos_a], axis=-1)],
            axis=-2,
        )
    elif axis == "y":
        mat = xp.stack(
            [xp.stack([cos_a, zero, sin_a], axis=-1), xp.stack([zero, ones, zero], axis=-1), xp.stack([-sin_a, zero, cos_a], axis=-1)],
            axis=-2,
        )
    elif axis == "z":
        mat = xp.stack(
            [xp.stack([cos_a, -sin_a, zero], axis=-1), xp.stack([sin_a, cos_a, zero], axis=-1), xp.stack([zero, zero, ones], axis=-1)],
            axis=-2,
        )
    else:
        raise ValueError(f"Invalid axis: {axis}")

    return mat


def _rotmat_to_euler(rotmat: Float[Any, "... 3 3"], convention: EulerConvention, *, xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    """Convert a rotation matrix to Euler angles."""
    assert convention in _EULER_CONVENTIONS, "Invalid Euler convention."
    if xp is None:
        xp = get_namespace(rotmat)

    is_extrinsic = convention.islower()
    angle_convention = convention.upper()[::-1] if is_extrinsic else convention
    eulers = _rotmat_to_euler_angles(rotmat, angle_convention, xp=xp)

    if is_extrinsic:
        return xp.stack([eulers[..., 2], eulers[..., 1], eulers[..., 0]], axis=-1)

    return eulers


def _rotmat_to_euler_angles(rotmat: Float[Any, "... 3 3"], convention: str, *, xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    """Extract Euler angles from a rotation matrix using systematic approach."""
    if xp is None:
        xp = get_namespace(rotmat)

    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2

    if tait_bryan:
        sign = xp.ones_like(rotmat[..., i0, i2])
        if i0 - i2 in [-1, 2]:
            sign = -sign
        x = rotmat[..., i0, i2] * sign

        one = xp.ones_like(x)
        eps = xp.asarray(common.safe_eps(x.dtype, xp, scale=1.0), dtype=x.dtype)
        central_angle = xp.arcsin(xp.clip(x, -one + eps, one - eps))
    else:
        one = xp.ones_like(rotmat[..., i0, i0])
        central_angle = xp.arccos(xp.clip(rotmat[..., i0, i0], -one, one))

    first_angle = _angle_from_tan(convention[0], convention[1], rotmat[..., i2], False, tait_bryan, xp)
    third_angle = _angle_from_tan(convention[2], convention[1], rotmat[..., i0, :], True, tait_bryan, xp)

    return xp.stack([first_angle, central_angle, third_angle], axis=-1)


def _angle_from_tan(
    axis: str, other_axis: str, data: Float[Any, "... 3"], horizontal: bool, tait_bryan: bool, xp: ModuleType
) -> Float[Any, "..."]:
    """Compute angle from tangent using systematic indexing."""
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return xp.arctan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return xp.arctan2(-data[..., i2], data[..., i1])
    return xp.arctan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str) -> int:
    """Convert axis letter to index."""
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")
