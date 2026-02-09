"""6D rotation representation conversions for SO(3)."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import common
from nanomanifold.common import get_namespace

from .matrix import from_matrix, to_matrix


def _normalize(vec: Float[Any, "... 3"], xp) -> Float[Any, "... 3"]:
    eps = common.safe_eps(vec.dtype, xp)
    norm = xp.linalg.norm(vec, axis=-1, keepdims=True)
    safe_norm = xp.where(norm < eps, xp.ones_like(norm), norm)
    return vec / safe_norm


def _cross(a: Float[Any, "... 3"], b: Float[Any, "... 3"], xp) -> Float[Any, "... 3"]:
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return xp.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1)


def to_6d(q: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    """Convert quaternion rotation to 6D representation.

    The 6D representation is formed by concatenating the first two columns of
    the rotation matrix.
    """
    if xp is None:
        xp = get_namespace(q)
    R = to_matrix(q, xp=xp)
    return xp.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def from_6d(d6: Float[Any, "... 6"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    """Convert 6D rotation representation to a quaternion.

    Applies Gram-Schmidt orthonormalization to ensure a valid rotation matrix.
    """
    if xp is None:
        xp = get_namespace(d6)
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]

    b1 = _normalize(a1, xp)
    dot = xp.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = _normalize(a2 - dot * b1, xp)
    b3 = _cross(b1, b2, xp)

    R = xp.stack([b1, b2, b3], axis=-1)
    return from_matrix(R, xp=xp)
