"""Conversions between ``(..., 7)`` ``[quaternion, translation]`` transforms, rotation-translation pairs and 4x4 matrices."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import SO3
from nanomanifold.common import get_namespace
from nanomanifold.SO3.primitives.quaternion import QuaternionConvention


def canonicalize(T: Float[Any, "... 7"], *, convention: QuaternionConvention = "wxyz", xp: ModuleType | None = None) -> Float[Any, "... 7"]:
    """Canonicalize the quaternion part of transforms ``(..., 7)``: unit norm, ``w >= 0``."""
    if xp is None:
        xp = get_namespace(T)
    return xp.concatenate([SO3.canonicalize(T[..., :4], convention=convention, xp=xp), T[..., 4:7]], axis=-1)


def from_rt(
    q: Float[Any, "... 4"],
    t: Float[Any, "... 3"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 7"]:
    """Transforms ``(..., 7)`` from rotations ``(..., 4)`` and translations ``(..., 3)``; the rotation is canonicalized."""
    if xp is None:
        xp = get_namespace(q)
    return canonicalize(xp.concatenate([q, t], axis=-1), convention=convention, xp=xp)


def to_rt(
    T: Float[Any, "... 7"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> tuple[Float[Any, "... 4"], Float[Any, "... 3"]]:
    """Canonical rotations ``(..., 4)`` and translations ``(..., 3)`` of transforms ``(..., 7)``."""
    if xp is None:
        xp = get_namespace(T)
    return SO3.canonicalize(T[..., :4], convention=convention, xp=xp), T[..., 4:7]


def to_matrix(T: Float[Any, "... 7"], *, convention: QuaternionConvention = "wxyz", xp: ModuleType | None = None) -> Float[Any, "... 4 4"]:
    """Homogeneous 4x4 matrices ``(..., 4, 4)`` of transforms ``(..., 7)``."""
    if xp is None:
        xp = get_namespace(T)
    rotmat = SO3.to_rotmat(T[..., :4], convention=convention, xp=xp)
    top_block = xp.concatenate([rotmat, T[..., 4:7, None]], axis=-1)
    zeros = xp.zeros_like(top_block[..., :1, :3])
    ones = xp.ones_like(top_block[..., :1, :1])
    return xp.concatenate([top_block, xp.concatenate([zeros, ones], axis=-1)], axis=-2)


def from_matrix(
    matrix: Float[Any, "... 4 4"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 7"]:
    """Transforms ``(..., 7)`` from homogeneous 4x4 matrices ``(..., 4, 4)`` whose top-left block is a rotation."""
    if xp is None:
        xp = get_namespace(matrix)
    q = SO3.from_rotmat(matrix[..., :3, :3], convention=convention, xp=xp)
    return xp.concatenate([q, matrix[..., :3, 3]], axis=-1)
