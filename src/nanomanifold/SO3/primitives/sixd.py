"""6D edges: sixd <-> rotmat. The 6D representation stacks the first two rotation-matrix columns."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import common


def sixd_to_rotmat(sixd: Float[Any, "... 6"], *, xp: ModuleType) -> Float[Any, "... 3 3"]:
    """Gram-Schmidt orthonormalization of the two stacked columns."""
    a1 = sixd[..., 0:3]
    a2 = sixd[..., 3:6]

    b1 = _normalize(a1, xp)
    dot = xp.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = _normalize(a2 - dot * b1, xp)
    b3 = _cross(b1, b2, xp)

    return xp.stack([b1, b2, b3], axis=-1)


def rotmat_to_sixd(rotmat: Float[Any, "... 3 3"], *, xp: ModuleType) -> Float[Any, "... 6"]:
    return xp.concatenate([rotmat[..., :, 0], rotmat[..., :, 1]], axis=-1)


def _normalize(vec: Float[Any, "... 3"], xp: ModuleType) -> Float[Any, "... 3"]:
    eps = common.safe_eps(vec.dtype, xp)
    norm = xp.linalg.norm(vec, axis=-1, keepdims=True)
    safe_norm = xp.where(norm < eps, xp.ones_like(norm), norm)
    return vec / safe_norm


def _cross(a: Float[Any, "... 3"], b: Float[Any, "... 3"], xp: ModuleType) -> Float[Any, "... 3"]:
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return xp.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1)
