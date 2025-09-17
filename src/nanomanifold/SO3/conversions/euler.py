from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace

from ..canonicalize import canonicalize
from . import matrix


def to_euler(q: Float[Any, "... 4"], convention: str = "ZYX") -> Float[Any, "... 3"]:
    q = canonicalize(q)
    R = matrix.to_matrix(q)
    return _matrix_to_euler(R, convention)


def from_euler(euler: Float[Any, "... 3"], convention: str = "ZYX") -> Float[Any, "... 4"]:
    R = _euler_to_matrix(euler, convention)
    return matrix.from_matrix(R)


def _euler_to_matrix(euler, convention):
    """Convert Euler angles to rotation matrix."""
    xp = get_namespace(euler)

    eye = xp.eye(3, dtype=euler.dtype)
    output_shape = euler.shape[:-1] + (3, 3)
    R = xp.broadcast_to(eye, output_shape)

    is_extrinsic = convention.islower()
    conv = convention.lower()

    for i, axis in enumerate(conv):
        angle = euler[..., i]
        R_axis = _rotation_matrix(angle, axis)

        if is_extrinsic:
            R = xp.matmul(R_axis, R)
        else:
            R = xp.matmul(R, R_axis)

    return R


def _rotation_matrix(angle, axis):
    """Create rotation matrix for given angle and axis."""
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


def _matrix_to_euler(matrix, convention):
    """Convert rotation matrix to Euler angles."""
    xp = get_namespace(matrix)

    is_extrinsic = convention.islower()

    if is_extrinsic:
        convention = convention.upper()
        convention = convention[::-1]

    eulers = _matrix_to_euler_angles(matrix, convention)

    if is_extrinsic:
        return xp.stack([eulers[..., 2], eulers[..., 1], eulers[..., 0]], axis=-1)

    return eulers


def _matrix_to_euler_angles(matrix, convention):
    """Extract Euler angles from rotation matrix using systematic approach."""
    xp = get_namespace(matrix)

    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")

    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2

    if tait_bryan:
        sign = -1.0 if i0 - i2 in [-1, 2] else 1.0
        x = matrix[..., i0, i2] * sign

        one = xp.ones_like(x)
        eps = xp.finfo(x.dtype).eps * one
        central_angle = xp.arcsin(xp.clip(x, -one + eps, one - eps))
    else:
        central_angle = xp.arccos(xp.clip(matrix[..., i0, i0], -1, 1))

    first_angle = _angle_from_tan(convention[0], convention[1], matrix[..., i2], False, tait_bryan, xp)
    third_angle = _angle_from_tan(convention[2], convention[1], matrix[..., i0, :], True, tait_bryan, xp)

    return xp.stack([first_angle, central_angle, third_angle], axis=-1)


def _angle_from_tan(axis, other_axis, data, horizontal, tait_bryan, xp):
    """Compute angle from tangent using systematic indexing."""
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return xp.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return xp.atan2(-data[..., i2], data[..., i1])
    return xp.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter):
    """Convert axis letter to index."""
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")
