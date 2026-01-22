"""Quaternion format conversions for SO(3)."""

from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace
from nanomanifold.SO3.canonicalize import canonicalize


def from_quat_xyzw(quat_xyzw: Float[Any, "... 4"]) -> Float[Any, "... 4"]:
    """Convert quaternion from [x, y, z, w] format to [w, x, y, z]."""
    xp = get_namespace(quat_xyzw)
    w = quat_xyzw[..., 3:4]
    xyz = quat_xyzw[..., 0:3]
    quat_wxyz = xp.concat([w, xyz], axis=-1)
    return canonicalize(quat_wxyz)


def to_quat_xyzw(quat_wxyz: Float[Any, "... 4"]) -> Float[Any, "... 4"]:
    """Convert quaternion from [w, x, y, z] format to [x, y, z, w]."""
    xp = get_namespace(quat_wxyz)
    quat_wxyz = canonicalize(quat_wxyz)
    w = quat_wxyz[..., 0:1]
    xyz = quat_wxyz[..., 1:4]
    return xp.concat([xyz, w], axis=-1)
