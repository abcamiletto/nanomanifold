"""Pairwise SO(3) representation conversions."""

import math
from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import common
from nanomanifold.common import get_namespace

from ..hat import hat as _hat
from ..identity import identity_as
from ..primitives.axis_angle import from_axis_angle as _from_axis_angle
from ..primitives.axis_angle import to_axis_angle as _to_axis_angle
from ..primitives.euler import _euler_to_rotmat, _rotmat_to_euler
from ..primitives.euler import from_euler as _from_euler
from ..primitives.euler import to_euler as _to_euler
from ..primitives.quaternion import canonicalize as _canonicalize
from ..primitives.quaternion import from_quat_xyzw as _from_quat_xyzw
from ..primitives.quaternion import to_quat_xyzw as _to_quat_xyzw
from ..primitives.rotmat import _project_matrix_to_rotmat
from ..primitives.rotmat import from_rotmat as _from_rotmat
from ..primitives.rotmat import to_rotmat as _to_rotmat
from ..primitives.sixd import _from_sixd_to_rotmat
from ..primitives.sixd import from_sixd as _from_sixd
from ..primitives.sixd import to_sixd as _to_sixd


def _axis_angle_to_rotmat_direct(axis_angle: Float[Any, "... 3"], xp) -> Float[Any, "... 3 3"]:
    theta = xp.linalg.norm(axis_angle, axis=-1, keepdims=True)
    theta2 = theta * theta
    one = xp.ones_like(theta)

    thresh = xp.asarray(math.sqrt(common.safe_eps(axis_angle.dtype, xp, scale=1.0)), dtype=axis_angle.dtype)
    small = theta < thresh
    safe_theta = xp.where(small, one, theta)
    safe_theta2 = safe_theta * safe_theta

    a_small = one - theta2 / 6.0
    b_small = 0.5 * one - theta2 / 24.0
    a = xp.where(small, a_small, xp.sin(theta) / safe_theta)
    b = xp.where(small, b_small, (one - xp.cos(theta)) / safe_theta2)

    K = _hat(axis_angle, xp=xp)
    K2 = xp.matmul(K, K)

    identity = identity_as(axis_angle, batch_dims=axis_angle.shape[:-1], rotation_type="rotmat", xp=xp)
    return identity + a[..., None] * K + b[..., None] * K2


def from_axis_angle_to_euler(
    axis_angle: Float[Any, "... 3"], *, convention: str = "ZYX", xp: ModuleType | None = None
) -> Float[Any, "... 3"]:
    return _to_euler(_from_axis_angle(axis_angle, xp=xp), convention=convention, xp=xp)


def from_axis_angle_to_rotmat(axis_angle: Float[Any, "... 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 3 3"]:
    if xp is None:
        xp = get_namespace(axis_angle)
    return _axis_angle_to_rotmat_direct(axis_angle, xp)


def from_axis_angle_to_quat_wxyz(axis_angle: Float[Any, "... 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return _canonicalize(_from_axis_angle(axis_angle, xp=xp), xp=xp)


def from_axis_angle_to_quat_xyzw(axis_angle: Float[Any, "... 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return _to_quat_xyzw(_from_axis_angle(axis_angle, xp=xp), xp=xp)


def from_axis_angle_to_sixd(axis_angle: Float[Any, "... 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    return from_rotmat_to_sixd(from_axis_angle_to_rotmat(axis_angle, xp=xp), xp=xp)


def from_euler_to_axis_angle(euler: Float[Any, "... 3"], *, convention: str = "ZYX", xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    return _to_axis_angle(_from_euler(euler, convention=convention, xp=xp), xp=xp)


def from_euler_to_euler(
    euler: Float[Any, "... 3"],
    *,
    source_convention: str = "ZYX",
    target_convention: str = "ZYX",
    xp: ModuleType | None = None,
) -> Float[Any, "... 3"]:
    return _to_euler(_from_euler(euler, convention=source_convention, xp=xp), convention=target_convention, xp=xp)


def from_euler_to_rotmat(euler: Float[Any, "... 3"], *, convention: str = "ZYX", xp: ModuleType | None = None) -> Float[Any, "... 3 3"]:
    return _euler_to_rotmat(euler, convention, xp=xp)


def from_euler_to_quat_wxyz(euler: Float[Any, "... 3"], *, convention: str = "ZYX", xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return _canonicalize(_from_euler(euler, convention=convention, xp=xp), xp=xp)


def from_euler_to_quat_xyzw(euler: Float[Any, "... 3"], *, convention: str = "ZYX", xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return _to_quat_xyzw(_from_euler(euler, convention=convention, xp=xp), xp=xp)


def from_euler_to_sixd(euler: Float[Any, "... 3"], *, convention: str = "ZYX", xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    return from_rotmat_to_sixd(from_euler_to_rotmat(euler, convention=convention, xp=xp), xp=xp)


def from_rotmat_to_axis_angle(rotmat: Float[Any, "... 3 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    return _to_axis_angle(_from_rotmat(rotmat, xp=xp), xp=xp)


def from_rotmat_to_euler(rotmat: Float[Any, "... 3 3"], *, convention: str = "ZYX", xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    return _rotmat_to_euler(rotmat, convention, xp=xp)


def from_rotmat_to_quat_wxyz(rotmat: Float[Any, "... 3 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return _from_rotmat(rotmat, xp=xp)


def from_rotmat_to_quat_xyzw(rotmat: Float[Any, "... 3 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return _to_quat_xyzw(_from_rotmat(rotmat, xp=xp), xp=xp)


def from_rotmat_to_sixd(rotmat: Float[Any, "... 3 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    if xp is None:
        xp = get_namespace(rotmat)
    return xp.concatenate([rotmat[..., :, 0], rotmat[..., :, 1]], axis=-1)


def from_matrix_to_rotmat(matrix: Float[Any, "... 3 3"], *, mode: str = "svd", xp: ModuleType | None = None) -> Float[Any, "... 3 3"]:
    if xp is None:
        xp = get_namespace(matrix)
    return _project_matrix_to_rotmat(matrix, xp, mode=mode)


def from_matrix_to_axis_angle(matrix: Float[Any, "... 3 3"], *, mode: str = "svd", xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    rotmat = from_matrix_to_rotmat(matrix, mode=mode, xp=xp)
    if xp is None:
        xp = get_namespace(rotmat)
    return from_rotmat_to_axis_angle(rotmat, xp=xp)


def from_matrix_to_euler(
    matrix: Float[Any, "... 3 3"],
    *,
    convention: str = "ZYX",
    mode: str = "svd",
    xp: ModuleType | None = None,
) -> Float[Any, "... 3"]:
    rotmat = from_matrix_to_rotmat(matrix, mode=mode, xp=xp)
    if xp is None:
        xp = get_namespace(rotmat)
    return from_rotmat_to_euler(rotmat, convention=convention, xp=xp)


def from_matrix_to_quat_wxyz(matrix: Float[Any, "... 3 3"], *, mode: str = "svd", xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    rotmat = from_matrix_to_rotmat(matrix, mode=mode, xp=xp)
    if xp is None:
        xp = get_namespace(rotmat)
    return from_rotmat_to_quat_wxyz(rotmat, xp=xp)


def from_matrix_to_quat_xyzw(matrix: Float[Any, "... 3 3"], *, mode: str = "svd", xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    rotmat = from_matrix_to_rotmat(matrix, mode=mode, xp=xp)
    if xp is None:
        xp = get_namespace(rotmat)
    return from_rotmat_to_quat_xyzw(rotmat, xp=xp)


def from_matrix_to_sixd(matrix: Float[Any, "... 3 3"], *, mode: str = "svd", xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    rotmat = from_matrix_to_rotmat(matrix, mode=mode, xp=xp)
    if xp is None:
        xp = get_namespace(rotmat)
    return from_rotmat_to_sixd(rotmat, xp=xp)


def from_quat_wxyz_to_axis_angle(quat_wxyz: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    return _to_axis_angle(quat_wxyz, xp=xp)


def from_quat_wxyz_to_euler(
    quat_wxyz: Float[Any, "... 4"], *, convention: str = "ZYX", xp: ModuleType | None = None
) -> Float[Any, "... 3"]:
    return _to_euler(quat_wxyz, convention=convention, xp=xp)


def from_quat_wxyz_to_rotmat(quat_wxyz: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 3 3"]:
    return _to_rotmat(quat_wxyz, xp=xp)


def from_quat_wxyz_to_quat_xyzw(quat_wxyz: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return _to_quat_xyzw(quat_wxyz, xp=xp)


def from_quat_wxyz_to_sixd(quat_wxyz: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    return _to_sixd(quat_wxyz, xp=xp)


def from_quat_xyzw_to_axis_angle(quat_xyzw: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    return _to_axis_angle(_from_quat_xyzw(quat_xyzw, xp=xp), xp=xp)


def from_quat_xyzw_to_euler(
    quat_xyzw: Float[Any, "... 4"], *, convention: str = "ZYX", xp: ModuleType | None = None
) -> Float[Any, "... 3"]:
    return _to_euler(_from_quat_xyzw(quat_xyzw, xp=xp), convention=convention, xp=xp)


def from_quat_xyzw_to_rotmat(quat_xyzw: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 3 3"]:
    return _to_rotmat(_from_quat_xyzw(quat_xyzw, xp=xp), xp=xp)


def from_quat_xyzw_to_quat_wxyz(quat_xyzw: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return _canonicalize(_from_quat_xyzw(quat_xyzw, xp=xp), xp=xp)


def from_quat_xyzw_to_sixd(quat_xyzw: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    return _to_sixd(_from_quat_xyzw(quat_xyzw, xp=xp), xp=xp)


def from_sixd_to_axis_angle(sixd: Float[Any, "... 6"], *, xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    return _to_axis_angle(_from_sixd(sixd, xp=xp), xp=xp)


def from_sixd_to_euler(sixd: Float[Any, "... 6"], *, convention: str = "ZYX", xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    return from_rotmat_to_euler(from_sixd_to_rotmat(sixd, xp=xp), convention=convention, xp=xp)


def from_sixd_to_rotmat(sixd: Float[Any, "... 6"], *, xp: ModuleType | None = None) -> Float[Any, "... 3 3"]:
    if xp is None:
        xp = get_namespace(sixd)
    return _from_sixd_to_rotmat(sixd, xp)


def from_sixd_to_quat_wxyz(sixd: Float[Any, "... 6"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return _canonicalize(_from_sixd(sixd, xp=xp), xp=xp)


def from_sixd_to_quat_xyzw(sixd: Float[Any, "... 6"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return _to_quat_xyzw(_from_sixd(sixd, xp=xp), xp=xp)


__all__ = [
    "from_axis_angle_to_euler",
    "from_axis_angle_to_rotmat",
    "from_axis_angle_to_quat_wxyz",
    "from_axis_angle_to_quat_xyzw",
    "from_axis_angle_to_sixd",
    "from_euler_to_axis_angle",
    "from_euler_to_euler",
    "from_euler_to_rotmat",
    "from_euler_to_quat_wxyz",
    "from_euler_to_quat_xyzw",
    "from_euler_to_sixd",
    "from_rotmat_to_axis_angle",
    "from_rotmat_to_euler",
    "from_rotmat_to_quat_wxyz",
    "from_rotmat_to_quat_xyzw",
    "from_rotmat_to_sixd",
    "from_matrix_to_rotmat",
    "from_matrix_to_axis_angle",
    "from_matrix_to_euler",
    "from_matrix_to_quat_wxyz",
    "from_matrix_to_quat_xyzw",
    "from_matrix_to_sixd",
    "from_quat_wxyz_to_axis_angle",
    "from_quat_wxyz_to_euler",
    "from_quat_wxyz_to_rotmat",
    "from_quat_wxyz_to_quat_xyzw",
    "from_quat_wxyz_to_sixd",
    "from_quat_xyzw_to_axis_angle",
    "from_quat_xyzw_to_euler",
    "from_quat_xyzw_to_rotmat",
    "from_quat_xyzw_to_quat_wxyz",
    "from_quat_xyzw_to_sixd",
    "from_sixd_to_axis_angle",
    "from_sixd_to_euler",
    "from_sixd_to_rotmat",
    "from_sixd_to_quat_wxyz",
    "from_sixd_to_quat_xyzw",
]
