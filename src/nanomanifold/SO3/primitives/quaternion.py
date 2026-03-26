"""Quaternion convention helpers for SO(3)."""

from types import ModuleType
from typing import Any, Literal

from jaxtyping import Float

from nanomanifold import common
from nanomanifold.common import get_namespace

QuaternionConvention = Literal["wxyz", "xyzw"]


def _to_wxyz(quat: Float[Any, "... 4"], *, convention: QuaternionConvention, xp) -> Float[Any, "... 4"]:
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if convention == "wxyz":
        return quat
    w = quat[..., 3:4]
    xyz = quat[..., 0:3]
    return xp.concatenate([w, xyz], axis=-1)


def _from_wxyz(quat: Float[Any, "... 4"], *, convention: QuaternionConvention, xp) -> Float[Any, "... 4"]:
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if convention == "wxyz":
        return quat
    w = quat[..., 0:1]
    xyz = quat[..., 1:4]
    return xp.concatenate([xyz, w], axis=-1)


def canonicalize(
    quat: Float[Any, "... 4"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace(quat)
    quat_wxyz = _to_wxyz(quat, convention=convention, xp=xp)

    norm = xp.sqrt(xp.sum(quat_wxyz**2, axis=-1, keepdims=True))
    eps = common.safe_eps(quat_wxyz.dtype, xp)
    safe_norm = xp.maximum(norm, xp.asarray(eps, dtype=quat_wxyz.dtype))
    quat_normalized = quat_wxyz / safe_norm

    mask = quat_normalized[..., 0:1] < 0
    quat_canonical = xp.where(mask, -quat_normalized, quat_normalized)

    return _from_wxyz(quat_canonical, convention=convention, xp=xp)


def from_quat(
    quat: Float[Any, "... 4"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    """Convert a quaternion from the given convention to canonical ``wxyz`` order."""
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace(quat)
    quat_wxyz = _to_wxyz(quat, convention=convention, xp=xp)
    return canonicalize(quat_wxyz, xp=xp)


def to_quat(
    quat: Float[Any, "... 4"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    """Convert a canonical ``wxyz`` quaternion to the requested convention."""
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace(quat)
    quat_wxyz = canonicalize(quat, xp=xp)
    return _from_wxyz(quat_wxyz, convention=convention, xp=xp)


__all__ = ["QuaternionConvention", "canonicalize", "from_quat", "to_quat"]
