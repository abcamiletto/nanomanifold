"""6D rotation representation conversions for SO(3)."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import common
from nanomanifold.common import get_namespace

from .quaternion import QuaternionConvention
from .rotmat import from_rotmat, to_rotmat


def _normalize(vec: Float[Any, "... 3"], xp) -> Float[Any, "... 3"]:
    eps = common.safe_eps(vec.dtype, xp)
    norm = xp.linalg.norm(vec, axis=-1, keepdims=True)
    safe_norm = xp.where(norm < eps, xp.ones_like(norm), norm)
    return vec / safe_norm


def _cross(a: Float[Any, "... 3"], b: Float[Any, "... 3"], xp) -> Float[Any, "... 3"]:
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return xp.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1)


def _from_sixd_to_rotmat(sixd: Float[Any, "... 6"], xp) -> Float[Any, "... 3 3"]:
    a1 = sixd[..., 0:3]
    a2 = sixd[..., 3:6]

    b1 = _normalize(a1, xp)
    dot = xp.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = _normalize(a2 - dot * b1, xp)
    b3 = _cross(b1, b2, xp)

    return xp.stack([b1, b2, b3], axis=-1)


def to_sixd(
    q: Float[Any, "... 4"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 6"]:
    """Convert a quaternion rotation to the 6D representation.

    The 6D representation is formed by concatenating the first two columns of
    the rotation matrix.
    """
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace(q)
    rotmat = to_rotmat(q, convention=convention, xp=xp)
    return xp.concatenate([rotmat[..., :, 0], rotmat[..., :, 1]], axis=-1)


def from_sixd(
    sixd: Float[Any, "... 6"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    """Convert the 6D rotation representation to a quaternion.

    Applies Gram-Schmidt orthonormalization to ensure a valid rotation matrix.
    """
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace(sixd)
    rotmat = _from_sixd_to_rotmat(sixd, xp)
    return from_rotmat(rotmat, convention=convention, xp=xp)
