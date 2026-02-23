"""Quaternion format conversions for SO(3)."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import common
from nanomanifold.common import get_namespace


def _from_quat_xyzw(quat_xyzw: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    if xp is None:
        xp = get_namespace(quat_xyzw)
    w = quat_xyzw[..., 3:4]
    xyz = quat_xyzw[..., 0:3]
    return xp.concatenate([w, xyz], axis=-1)


def _to_quat_xyzw(quat_wxyz: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    if xp is None:
        xp = get_namespace(quat_wxyz)
    w = quat_wxyz[..., 0:1]
    xyz = quat_wxyz[..., 1:4]
    return xp.concatenate([xyz, w], axis=-1)


def canonicalize(q: Float[Any, "... 4"], xyzw: bool = False, *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    if xp is None:
        xp = get_namespace(q)
    if xyzw:
        q = _from_quat_xyzw(q, xp=xp)

    norm = xp.sqrt(xp.sum(q**2, axis=-1, keepdims=True))
    eps = common.safe_eps(q.dtype, xp)
    safe_norm = xp.maximum(norm, xp.asarray(eps, dtype=q.dtype))
    q_normalized = q / safe_norm

    mask = q_normalized[..., 0:1] < 0
    q_canonical = xp.where(mask, -q_normalized, q_normalized)

    if xyzw:
        return _to_quat_xyzw(q_canonical, xp=xp)
    return q_canonical


def from_quat_xyzw(quat_xyzw: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    """Convert quaternion from [x, y, z, w] format to [w, x, y, z]."""
    quat_wxyz = _from_quat_xyzw(quat_xyzw, xp=xp)
    return canonicalize(quat_wxyz, xp=xp)


def to_quat_xyzw(quat_wxyz: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    """Convert quaternion from [w, x, y, z] format to [x, y, z, w]."""
    quat_wxyz = canonicalize(quat_wxyz, xp=xp)
    return _to_quat_xyzw(quat_wxyz, xp=xp)
