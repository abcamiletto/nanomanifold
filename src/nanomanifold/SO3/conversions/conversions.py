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
from ..primitives.euler import _EULER_CONVENTIONS, EulerConvention, _euler_to_rotmat, _rotmat_to_euler
from ..primitives.euler import from_euler as _from_euler
from ..primitives.euler import to_euler as _to_euler
from ..primitives.quaternion import QuaternionConvention
from ..primitives.quaternion import canonicalize as _canonicalize
from ..primitives.quaternion import from_quat as _from_quat
from ..primitives.quaternion import to_quat as _to_quat
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
    axis_angle: Float[Any, "... 3"], *, convention: EulerConvention = "ZYX", xp: ModuleType | None = None
) -> Float[Any, "... 3"]:
    assert convention in _EULER_CONVENTIONS, "Invalid Euler convention."
    return _to_euler(_from_axis_angle(axis_angle, xp=xp), convention=convention, xp=xp)


def from_axis_angle_to_hinge(
    axis_angle: Float[Any, "... 3"],
    axes: Float[Any, "... 3"],
    *,
    xp: ModuleType | None = None,
) -> Float[Any, "... 1"]:
    if xp is None:
        xp = get_namespace(axis_angle)
    projected = xp.sum(axis_angle * axes, axis=-1, keepdims=True)
    axis_norm_sq = xp.sum(axes * axes, axis=-1, keepdims=True)
    return projected / axis_norm_sq


def from_axis_angle_to_rotmat(axis_angle: Float[Any, "... 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 3 3"]:
    if xp is None:
        xp = get_namespace(axis_angle)
    return _axis_angle_to_rotmat_direct(axis_angle, xp)


def from_axis_angle_to_quat(
    axis_angle: Float[Any, "... 3"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    return _to_quat(_from_axis_angle(axis_angle, xp=xp), convention=convention, xp=xp)


def from_axis_angle_to_sixd(axis_angle: Float[Any, "... 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    return from_rotmat_to_sixd(from_axis_angle_to_rotmat(axis_angle, xp=xp), xp=xp)


def from_euler_to_axis_angle(
    euler: Float[Any, "... 3"], *, convention: EulerConvention = "ZYX", xp: ModuleType | None = None
) -> Float[Any, "... 3"]:
    assert convention in _EULER_CONVENTIONS, "Invalid Euler convention."
    return _to_axis_angle(_from_euler(euler, convention=convention, xp=xp), xp=xp)


def from_euler_to_hinge(
    euler: Float[Any, "... 3"],
    axes: Float[Any, "... 3"],
    *,
    convention: EulerConvention = "ZYX",
    xp: ModuleType | None = None,
) -> Float[Any, "... 1"]:
    assert convention in _EULER_CONVENTIONS, "Invalid Euler convention."
    axis_angle = from_euler_to_axis_angle(euler, convention=convention, xp=xp)
    return from_axis_angle_to_hinge(axis_angle, axes, xp=xp)


def from_euler_to_euler(
    euler: Float[Any, "... 3"],
    *,
    src_convention: EulerConvention = "ZYX",
    dst_convention: EulerConvention = "ZYX",
    xp: ModuleType | None = None,
) -> Float[Any, "... 3"]:
    assert src_convention in _EULER_CONVENTIONS, "Invalid Euler convention."
    assert dst_convention in _EULER_CONVENTIONS, "Invalid Euler convention."
    return _to_euler(_from_euler(euler, convention=src_convention, xp=xp), convention=dst_convention, xp=xp)


def from_euler_to_rotmat(
    euler: Float[Any, "... 3"], *, convention: EulerConvention = "ZYX", xp: ModuleType | None = None
) -> Float[Any, "... 3 3"]:
    assert convention in _EULER_CONVENTIONS, "Invalid Euler convention."
    return _euler_to_rotmat(euler, convention, xp=xp)


def from_euler_to_quat(
    euler: Float[Any, "... 3"],
    *,
    src_convention: EulerConvention = "ZYX",
    dst_convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    assert src_convention in _EULER_CONVENTIONS, "Invalid Euler convention."
    assert dst_convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    return _to_quat(_from_euler(euler, convention=src_convention, xp=xp), convention=dst_convention, xp=xp)


def from_euler_to_sixd(
    euler: Float[Any, "... 3"], *, convention: EulerConvention = "ZYX", xp: ModuleType | None = None
) -> Float[Any, "... 6"]:
    assert convention in _EULER_CONVENTIONS, "Invalid Euler convention."
    return from_rotmat_to_sixd(from_euler_to_rotmat(euler, convention=convention, xp=xp), xp=xp)


def from_hinge_to_axis_angle(
    angles: Float[Any, "... 1"],
    axes: Float[Any, "... 3"],
    *,
    xp: ModuleType | None = None,
) -> Float[Any, "... 3"]:
    assert angles.shape[-1:] == (1,), "Hinge angles must have shape (..., 1)."
    return angles * axes


def from_hinge_to_euler(
    angles: Float[Any, "... 1"],
    axes: Float[Any, "... 3"],
    *,
    convention: EulerConvention = "ZYX",
    xp: ModuleType | None = None,
) -> Float[Any, "... 3"]:
    assert convention in _EULER_CONVENTIONS, "Invalid Euler convention."
    axis_angle = from_hinge_to_axis_angle(angles, axes, xp=xp)
    return from_axis_angle_to_euler(axis_angle, convention=convention, xp=xp)


def from_hinge_to_rotmat(
    angles: Float[Any, "... 1"],
    axes: Float[Any, "... 3"],
    *,
    xp: ModuleType | None = None,
) -> Float[Any, "... 3 3"]:
    axis_angle = from_hinge_to_axis_angle(angles, axes, xp=xp)
    return from_axis_angle_to_rotmat(axis_angle, xp=xp)


def from_hinge_to_quat(
    angles: Float[Any, "... 1"],
    axes: Float[Any, "... 3"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    axis_angle = from_hinge_to_axis_angle(angles, axes, xp=xp)
    return from_axis_angle_to_quat(axis_angle, convention=convention, xp=xp)


def from_hinge_to_sixd(
    angles: Float[Any, "... 1"],
    axes: Float[Any, "... 3"],
    *,
    xp: ModuleType | None = None,
) -> Float[Any, "... 6"]:
    return from_rotmat_to_sixd(from_hinge_to_rotmat(angles, axes, xp=xp), xp=xp)


def from_rotmat_to_axis_angle(rotmat: Float[Any, "... 3 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    return _to_axis_angle(_from_rotmat(rotmat, xp=xp), xp=xp)


def from_rotmat_to_hinge(
    rotmat: Float[Any, "... 3 3"],
    axes: Float[Any, "... 3"],
    *,
    xp: ModuleType | None = None,
) -> Float[Any, "... 1"]:
    axis_angle = from_rotmat_to_axis_angle(rotmat, xp=xp)
    return from_axis_angle_to_hinge(axis_angle, axes, xp=xp)


def from_rotmat_to_euler(
    rotmat: Float[Any, "... 3 3"], *, convention: EulerConvention = "ZYX", xp: ModuleType | None = None
) -> Float[Any, "... 3"]:
    assert convention in _EULER_CONVENTIONS, "Invalid Euler convention."
    return _rotmat_to_euler(rotmat, convention, xp=xp)


def from_rotmat_to_quat(
    rotmat: Float[Any, "... 3 3"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    return _to_quat(_from_rotmat(rotmat, xp=xp), convention=convention, xp=xp)


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
    return from_rotmat_to_axis_angle(rotmat, xp=xp)


def from_matrix_to_hinge(
    matrix: Float[Any, "... 3 3"],
    axes: Float[Any, "... 3"],
    *,
    mode: str = "svd",
    xp: ModuleType | None = None,
) -> Float[Any, "... 1"]:
    axis_angle = from_matrix_to_axis_angle(matrix, mode=mode, xp=xp)
    return from_axis_angle_to_hinge(axis_angle, axes, xp=xp)


def from_matrix_to_euler(
    matrix: Float[Any, "... 3 3"],
    *,
    convention: EulerConvention = "ZYX",
    mode: str = "svd",
    xp: ModuleType | None = None,
) -> Float[Any, "... 3"]:
    assert convention in _EULER_CONVENTIONS, "Invalid Euler convention."
    rotmat = from_matrix_to_rotmat(matrix, mode=mode, xp=xp)
    return from_rotmat_to_euler(rotmat, convention=convention, xp=xp)


def from_matrix_to_quat(
    matrix: Float[Any, "... 3 3"],
    *,
    convention: QuaternionConvention = "wxyz",
    mode: str = "svd",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    rotmat = from_matrix_to_rotmat(matrix, mode=mode, xp=xp)
    return from_rotmat_to_quat(rotmat, convention=convention, xp=xp)


def from_matrix_to_sixd(matrix: Float[Any, "... 3 3"], *, mode: str = "svd", xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    rotmat = from_matrix_to_rotmat(matrix, mode=mode, xp=xp)
    return from_rotmat_to_sixd(rotmat, xp=xp)


def from_quat_to_axis_angle(
    quat: Float[Any, "... 4"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 3"]:
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    return _to_axis_angle(_from_quat(quat, convention=convention, xp=xp), xp=xp)


def from_quat_to_hinge(
    quat: Float[Any, "... 4"],
    axes: Float[Any, "... 3"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 1"]:
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    axis_angle = from_quat_to_axis_angle(quat, convention=convention, xp=xp)
    return from_axis_angle_to_hinge(axis_angle, axes, xp=xp)


def from_quat_to_euler(
    quat: Float[Any, "... 4"],
    *,
    src_convention: QuaternionConvention = "wxyz",
    dst_convention: EulerConvention = "ZYX",
    xp: ModuleType | None = None,
) -> Float[Any, "... 3"]:
    assert src_convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    assert dst_convention in _EULER_CONVENTIONS, "Invalid Euler convention."
    quat_wxyz = _from_quat(quat, convention=src_convention, xp=xp)
    return _to_euler(quat_wxyz, convention=dst_convention, xp=xp)


def from_quat_to_rotmat(
    quat: Float[Any, "... 4"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 3 3"]:
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    return _to_rotmat(_from_quat(quat, convention=convention, xp=xp), xp=xp)


def from_quat_to_quat(
    quat: Float[Any, "... 4"],
    *,
    src_convention: QuaternionConvention = "wxyz",
    dst_convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    assert src_convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    assert dst_convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    quat_wxyz = _canonicalize(_from_quat(quat, convention=src_convention, xp=xp), xp=xp)
    return _to_quat(quat_wxyz, convention=dst_convention, xp=xp)


def from_quat_to_sixd(
    quat: Float[Any, "... 4"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 6"]:
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    return _to_sixd(_from_quat(quat, convention=convention, xp=xp), xp=xp)


def from_sixd_to_axis_angle(sixd: Float[Any, "... 6"], *, xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    return _to_axis_angle(_from_sixd(sixd, xp=xp), xp=xp)


def from_sixd_to_hinge(
    sixd: Float[Any, "... 6"],
    axes: Float[Any, "... 3"],
    *,
    xp: ModuleType | None = None,
) -> Float[Any, "... 1"]:
    axis_angle = from_sixd_to_axis_angle(sixd, xp=xp)
    return from_axis_angle_to_hinge(axis_angle, axes, xp=xp)


def from_sixd_to_euler(
    sixd: Float[Any, "... 6"], *, convention: EulerConvention = "ZYX", xp: ModuleType | None = None
) -> Float[Any, "... 3"]:
    assert convention in _EULER_CONVENTIONS, "Invalid Euler convention."
    return from_rotmat_to_euler(from_sixd_to_rotmat(sixd, xp=xp), convention=convention, xp=xp)


def from_sixd_to_rotmat(sixd: Float[Any, "... 6"], *, xp: ModuleType | None = None) -> Float[Any, "... 3 3"]:
    if xp is None:
        xp = get_namespace(sixd)
    return _from_sixd_to_rotmat(sixd, xp)


def from_sixd_to_quat(
    sixd: Float[Any, "... 6"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    return _to_quat(_from_sixd(sixd, xp=xp), convention=convention, xp=xp)


__all__ = [
    "from_axis_angle_to_euler",
    "from_axis_angle_to_hinge",
    "from_axis_angle_to_rotmat",
    "from_axis_angle_to_quat",
    "from_axis_angle_to_sixd",
    "from_euler_to_axis_angle",
    "from_euler_to_hinge",
    "from_euler_to_euler",
    "from_euler_to_rotmat",
    "from_euler_to_quat",
    "from_euler_to_sixd",
    "from_hinge_to_axis_angle",
    "from_hinge_to_euler",
    "from_hinge_to_rotmat",
    "from_hinge_to_quat",
    "from_hinge_to_sixd",
    "from_rotmat_to_axis_angle",
    "from_rotmat_to_hinge",
    "from_rotmat_to_euler",
    "from_rotmat_to_quat",
    "from_rotmat_to_sixd",
    "from_matrix_to_rotmat",
    "from_matrix_to_axis_angle",
    "from_matrix_to_hinge",
    "from_matrix_to_euler",
    "from_matrix_to_quat",
    "from_matrix_to_sixd",
    "from_quat_to_axis_angle",
    "from_quat_to_hinge",
    "from_quat_to_euler",
    "from_quat_to_rotmat",
    "from_quat_to_quat",
    "from_quat_to_sixd",
    "from_sixd_to_axis_angle",
    "from_sixd_to_hinge",
    "from_sixd_to_euler",
    "from_sixd_to_rotmat",
    "from_sixd_to_quat",
]
