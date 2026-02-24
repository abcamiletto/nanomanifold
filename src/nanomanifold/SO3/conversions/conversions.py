"""Pairwise SO(3) representation conversions."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import common
from nanomanifold.common import get_namespace

from ..hat import hat as _hat
from ..primitives.axis_angle import from_axis_angle as _from_axis_angle
from ..primitives.axis_angle import to_axis_angle as _to_axis_angle
from ..primitives.euler import _euler_to_matrix, _matrix_to_euler
from ..primitives.euler import from_euler as _from_euler
from ..primitives.euler import to_euler as _to_euler
from ..primitives.matrix import from_matrix as _from_matrix
from ..primitives.matrix import to_matrix as _to_matrix
from ..primitives.quaternion import canonicalize as _canonicalize
from ..primitives.quaternion import from_quat_xyzw as _from_quat_xyzw
from ..primitives.quaternion import to_quat_xyzw as _to_quat_xyzw
from ..primitives.sixd import _from_6d_to_matrix
from ..primitives.sixd import from_6d as _from_6d
from ..primitives.sixd import to_6d as _to_6d


def _axis_angle_to_matrix_direct(axis_angle: Float[Any, "... 3"], xp) -> Float[Any, "... 3 3"]:
    theta = xp.linalg.norm(axis_angle, axis=-1, keepdims=True)
    theta2 = theta * theta
    one = xp.ones_like(theta)

    thresh = xp.asarray(common.small_angle_threshold(axis_angle.dtype, xp), dtype=axis_angle.dtype)
    small = theta < thresh
    safe_theta = xp.where(small, one, theta)
    safe_theta2 = safe_theta * safe_theta

    a_small = one - theta2 / 6.0
    b_small = 0.5 * one - theta2 / 24.0
    a = xp.where(small, a_small, xp.sin(theta) / safe_theta)
    b = xp.where(small, b_small, (one - xp.cos(theta)) / safe_theta2)

    K = _hat(axis_angle, xp=xp)
    K2 = xp.matmul(K, K)

    I = common.eye(3, dtype=axis_angle.dtype, xp=xp, like=axis_angle)
    I = xp.broadcast_to(I, axis_angle.shape[:-1] + (3, 3))
    return I + a[..., None] * K + b[..., None] * K2


def from_axis_angle_to_euler(
    axis_angle: Float[Any, "... 3"], *, convention: str = "ZYX", xp: ModuleType | None = None
) -> Float[Any, "... 3"]:
    return _to_euler(_from_axis_angle(axis_angle, xp=xp), convention=convention, xp=xp)


def from_axis_angle_to_matrix(axis_angle: Float[Any, "... 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 3 3"]:
    if xp is None:
        xp = get_namespace(axis_angle)
    return _axis_angle_to_matrix_direct(axis_angle, xp)


def from_axis_angle_to_quat_wxyz(axis_angle: Float[Any, "... 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return _canonicalize(_from_axis_angle(axis_angle, xp=xp), xp=xp)


def from_axis_angle_to_quat_xyzw(axis_angle: Float[Any, "... 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return _to_quat_xyzw(_from_axis_angle(axis_angle, xp=xp), xp=xp)


def from_axis_angle_to_sixd(axis_angle: Float[Any, "... 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    if xp is None:
        xp = get_namespace(axis_angle)
    matrix = _axis_angle_to_matrix_direct(axis_angle, xp)
    return from_matrix_to_sixd(matrix, xp=xp)


def from_euler_to_axis_angle(euler: Float[Any, "... 3"], *, convention: str = "ZYX", xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    return _to_axis_angle(_from_euler(euler, convention=convention, xp=xp), xp=xp)


def from_euler_to_matrix(euler: Float[Any, "... 3"], *, convention: str = "ZYX", xp: ModuleType | None = None) -> Float[Any, "... 3 3"]:
    return _euler_to_matrix(euler, convention, xp=xp)


def from_euler_to_quat_wxyz(euler: Float[Any, "... 3"], *, convention: str = "ZYX", xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return _canonicalize(_from_euler(euler, convention=convention, xp=xp), xp=xp)


def from_euler_to_quat_xyzw(euler: Float[Any, "... 3"], *, convention: str = "ZYX", xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return _to_quat_xyzw(_from_euler(euler, convention=convention, xp=xp), xp=xp)


def from_euler_to_sixd(euler: Float[Any, "... 3"], *, convention: str = "ZYX", xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    return from_matrix_to_sixd(from_euler_to_matrix(euler, convention=convention, xp=xp), xp=xp)


def from_matrix_to_axis_angle(matrix: Float[Any, "... 3 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    return _to_axis_angle(_from_matrix(matrix, xp=xp), xp=xp)


def from_matrix_to_euler(matrix: Float[Any, "... 3 3"], *, convention: str = "ZYX", xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    return _matrix_to_euler(matrix, convention, xp=xp)


def from_matrix_to_quat_wxyz(matrix: Float[Any, "... 3 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return _canonicalize(_from_matrix(matrix, xp=xp), xp=xp)


def from_matrix_to_quat_xyzw(matrix: Float[Any, "... 3 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return _to_quat_xyzw(_from_matrix(matrix, xp=xp), xp=xp)


def from_matrix_to_sixd(matrix: Float[Any, "... 3 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    if xp is None:
        xp = get_namespace(matrix)
    return xp.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def from_quat_wxyz_to_axis_angle(quat_wxyz: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    return _to_axis_angle(quat_wxyz, xp=xp)


def from_quat_wxyz_to_euler(
    quat_wxyz: Float[Any, "... 4"], *, convention: str = "ZYX", xp: ModuleType | None = None
) -> Float[Any, "... 3"]:
    return _to_euler(quat_wxyz, convention=convention, xp=xp)


def from_quat_wxyz_to_matrix(quat_wxyz: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 3 3"]:
    return _to_matrix(quat_wxyz, xp=xp)


def from_quat_wxyz_to_quat_xyzw(quat_wxyz: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return _to_quat_xyzw(quat_wxyz, xp=xp)


def from_quat_wxyz_to_sixd(quat_wxyz: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    return _to_6d(quat_wxyz, xp=xp)


def from_quat_xyzw_to_axis_angle(quat_xyzw: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    return _to_axis_angle(_from_quat_xyzw(quat_xyzw, xp=xp), xp=xp)


def from_quat_xyzw_to_euler(
    quat_xyzw: Float[Any, "... 4"], *, convention: str = "ZYX", xp: ModuleType | None = None
) -> Float[Any, "... 3"]:
    return _to_euler(_from_quat_xyzw(quat_xyzw, xp=xp), convention=convention, xp=xp)


def from_quat_xyzw_to_matrix(quat_xyzw: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 3 3"]:
    return _to_matrix(_from_quat_xyzw(quat_xyzw, xp=xp), xp=xp)


def from_quat_xyzw_to_quat_wxyz(quat_xyzw: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return _canonicalize(_from_quat_xyzw(quat_xyzw, xp=xp), xp=xp)


def from_quat_xyzw_to_sixd(quat_xyzw: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    return _to_6d(_from_quat_xyzw(quat_xyzw, xp=xp), xp=xp)


def from_sixd_to_axis_angle(sixd: Float[Any, "... 6"], *, xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    return _to_axis_angle(_from_6d(sixd, xp=xp), xp=xp)


def from_sixd_to_euler(sixd: Float[Any, "... 6"], *, convention: str = "ZYX", xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    return from_matrix_to_euler(from_sixd_to_matrix(sixd, xp=xp), convention=convention, xp=xp)


def from_sixd_to_matrix(sixd: Float[Any, "... 6"], *, xp: ModuleType | None = None) -> Float[Any, "... 3 3"]:
    if xp is None:
        xp = get_namespace(sixd)
    return _from_6d_to_matrix(sixd, xp)


def from_sixd_to_quat_wxyz(sixd: Float[Any, "... 6"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return _canonicalize(_from_6d(sixd, xp=xp), xp=xp)


def from_sixd_to_quat_xyzw(sixd: Float[Any, "... 6"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return _to_quat_xyzw(_from_6d(sixd, xp=xp), xp=xp)


__all__ = [
    "from_axis_angle_to_euler",
    "from_axis_angle_to_matrix",
    "from_axis_angle_to_quat_wxyz",
    "from_axis_angle_to_quat_xyzw",
    "from_axis_angle_to_sixd",
    "from_euler_to_axis_angle",
    "from_euler_to_matrix",
    "from_euler_to_quat_wxyz",
    "from_euler_to_quat_xyzw",
    "from_euler_to_sixd",
    "from_matrix_to_axis_angle",
    "from_matrix_to_euler",
    "from_matrix_to_quat_wxyz",
    "from_matrix_to_quat_xyzw",
    "from_matrix_to_sixd",
    "from_quat_wxyz_to_axis_angle",
    "from_quat_wxyz_to_euler",
    "from_quat_wxyz_to_matrix",
    "from_quat_wxyz_to_quat_xyzw",
    "from_quat_wxyz_to_sixd",
    "from_quat_xyzw_to_axis_angle",
    "from_quat_xyzw_to_euler",
    "from_quat_xyzw_to_matrix",
    "from_quat_xyzw_to_quat_wxyz",
    "from_quat_xyzw_to_sixd",
    "from_sixd_to_axis_angle",
    "from_sixd_to_euler",
    "from_sixd_to_matrix",
    "from_sixd_to_quat_wxyz",
    "from_sixd_to_quat_xyzw",
]
