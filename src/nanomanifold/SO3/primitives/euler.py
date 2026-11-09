"""Euler edges: euler <-> rotmat.

Uppercase conventions are intrinsic (rotations about the moving axes), lowercase are extrinsic
(rotations about the fixed axes).
"""

from types import ModuleType
from typing import Any, Literal, get_args

from jaxtyping import Float

from nanomanifold import common

EulerConvention = Literal[
    "xyz", "xzy", "yxz", "yzx", "zxy", "zyx",
    "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX",
    "xyx", "xzx", "yxy", "yzy", "zxz", "zyz",
    "XYX", "XZX", "YXY", "YZY", "ZXZ", "ZYZ",
]  # fmt: skip

EULER_CONVENTIONS = get_args(EulerConvention)


def euler_to_rotmat(euler: Float[Any, "... 3"], *, convention: EulerConvention = "ZYX", xp: ModuleType) -> Float[Any, "... 3 3"]:
    assert convention in EULER_CONVENTIONS, "Invalid Euler convention."
    first, second, third = [_axis_rotmat(euler[..., i], axis, xp) for i, axis in enumerate(convention.lower())]
    if convention.islower():
        return xp.matmul(third, xp.matmul(second, first))
    return xp.matmul(first, xp.matmul(second, third))


def rotmat_to_euler(rotmat: Float[Any, "... 3 3"], *, convention: EulerConvention = "ZYX", xp: ModuleType) -> Float[Any, "... 3"]:
    assert convention in EULER_CONVENTIONS, "Invalid Euler convention."
    if convention.isupper():
        return _rotmat_to_intrinsic_angles(rotmat, convention, xp)
    # An extrinsic sequence equals the reversed intrinsic sequence with the angles reversed.
    angles = _rotmat_to_intrinsic_angles(rotmat, convention.upper()[::-1], xp)
    return xp.stack([angles[..., 2], angles[..., 1], angles[..., 0]], axis=-1)


def _axis_rotmat(angle: Float[Any, "..."], axis: str, xp: ModuleType) -> Float[Any, "... 3 3"]:
    cos = xp.cos(angle)
    sin = xp.sin(angle)
    zero = xp.zeros_like(cos)
    one = xp.ones_like(cos)
    if axis == "x":
        rows = [[one, zero, zero], [zero, cos, -sin], [zero, sin, cos]]
    elif axis == "y":
        rows = [[cos, zero, sin], [zero, one, zero], [-sin, zero, cos]]
    else:
        rows = [[cos, -sin, zero], [sin, cos, zero], [zero, zero, one]]
    return xp.stack([xp.stack(row, axis=-1) for row in rows], axis=-2)


def _rotmat_to_intrinsic_angles(rotmat: Float[Any, "... 3 3"], convention: str, xp: ModuleType) -> Float[Any, "... 3"]:
    i0 = "XYZ".index(convention[0])
    i2 = "XYZ".index(convention[2])
    tait_bryan = i0 != i2

    if tait_bryan:
        x = rotmat[..., i0, i2]
        if i0 - i2 in (-1, 2):
            x = -x
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
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return xp.arctan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return xp.arctan2(-data[..., i2], data[..., i1])
    return xp.arctan2(data[..., i2], -data[..., i1])
