"""Matrix conversions for SE(3) transformations."""

from typing import Any

from jaxtyping import Float

from nanomanifold import SO3
from nanomanifold.common import get_namespace

from ..canonicalize import canonicalize


def to_matrix(se3: Float[Any, "... 7"], xyzw: bool = False) -> Float[Any, "... 4 4"]:
    """Convert SE(3) representation to 4x4 transformation matrix.

    Args:
        se3: SE(3) representation (..., 7) as [w, x, y, z, tx, ty, tz]
            or [x, y, z, w, tx, ty, tz] if xyzw=True
        xyzw: Whether to interpret the quaternion as [x, y, z, w]

    Returns:
        4x4 transformation matrix (..., 4, 4)
    """
    xp = get_namespace(se3)

    quat = se3[..., :4]
    translation = se3[..., 4:7]

    R = SO3.to_matrix(quat, xyzw=xyzw)

    translation_column = translation[..., None]
    top_block = xp.concatenate([R, translation_column], axis=-1)

    zeros = xp.zeros(top_block.shape[:-2] + (1, 3), dtype=se3.dtype)
    ones = xp.ones(top_block.shape[:-2] + (1, 1), dtype=se3.dtype)
    bottom_row = xp.concatenate([zeros, ones], axis=-1)

    return xp.concatenate([top_block, bottom_row], axis=-2)


def from_matrix(matrix: Float[Any, "... 4 4"]) -> Float[Any, "... 7"]:
    """Convert 4x4 transformation matrix to SE(3) representation.

    Args:
        matrix: 4x4 transformation matrix (..., 4, 4)

    Returns:
        SE(3) representation (..., 7) as [w, x, y, z, tx, ty, tz]
    """
    xp = get_namespace(matrix)

    R = matrix[..., :3, :3]

    quat = SO3.from_matrix(R)

    translation = matrix[..., :3, 3]

    se3 = xp.concatenate([quat, translation], axis=-1)
    return canonicalize(se3)
