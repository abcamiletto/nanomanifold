"""Matrix conversions for SE(3) transformations."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import SO3
from nanomanifold.common import get_namespace

from ..canonicalize import canonicalize


def to_matrix(se3: Float[Any, "... 7"], xyzw: bool = False, *, xp: ModuleType | None = None) -> Float[Any, "... 4 4"]:
    """Convert SE(3) representation to 4x4 transformation matrix.

    Args:
        se3: SE(3) representation (..., 7) as [w, x, y, z, tx, ty, tz]
            or [x, y, z, w, tx, ty, tz] if xyzw=True
        xyzw: Whether to interpret the quaternion as [x, y, z, w]
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        4x4 transformation matrix (..., 4, 4)
    """
    if xp is None:
        xp = get_namespace(se3)

    quat = se3[..., :4]
    translation = se3[..., 4:7]

    R = SO3.to_matrix(quat, xyzw=xyzw, xp=xp)

    translation_column = translation[..., None]
    top_block = xp.concatenate([R, translation_column], axis=-1)

    zeros = xp.zeros_like(top_block[..., :1, :3])
    ones = xp.ones_like(top_block[..., :1, :1])
    bottom_row = xp.concatenate([zeros, ones], axis=-1)

    return xp.concatenate([top_block, bottom_row], axis=-2)


def from_matrix(matrix: Float[Any, "... 4 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 7"]:
    """Convert 4x4 transformation matrix to SE(3) representation.

    Args:
        matrix: 4x4 transformation matrix (..., 4, 4)
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        SE(3) representation (..., 7) as [w, x, y, z, tx, ty, tz]
    """
    if xp is None:
        xp = get_namespace(matrix)

    R = matrix[..., :3, :3]

    quat = SO3.from_matrix(R, xp=xp)

    translation = matrix[..., :3, 3]

    se3 = xp.concatenate([quat, translation], axis=-1)
    return canonicalize(se3, xp=xp)
