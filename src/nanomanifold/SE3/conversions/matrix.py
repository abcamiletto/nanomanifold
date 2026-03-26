"""Matrix conversions for SE(3) transformations."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import SO3
from nanomanifold.common import get_namespace
from nanomanifold.SO3.primitives.quaternion import QuaternionConvention

from ..canonicalize import canonicalize


def to_matrix(
    se3: Float[Any, "... 7"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4 4"]:
    """Convert SE(3) representation to 4x4 transformation matrix.

    Args:
        se3: SE(3) representation (..., 7) with quaternion components in the given convention
        convention: Quaternion component order, either ``"wxyz"`` or ``"xyzw"``
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        4x4 transformation matrix (..., 4, 4)
    """
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace(se3)

    quat = se3[..., :4]
    translation = se3[..., 4:7]

    rotmat = SO3.to_rotmat(quat, convention=convention, xp=xp)

    translation_column = translation[..., None]
    top_block = xp.concatenate([rotmat, translation_column], axis=-1)

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
        SE(3) representation (..., 7) with quaternion components in ``wxyz`` order
    """
    if xp is None:
        xp = get_namespace(matrix)

    rotmat = matrix[..., :3, :3]
    quat = SO3.from_rotmat(rotmat, xp=xp)
    translation = matrix[..., :3, 3]

    se3 = xp.concatenate([quat, translation], axis=-1)
    return canonicalize(se3, xp=xp)
