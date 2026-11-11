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
    b3 = common.cross(b1, b2, xp)

    return xp.stack([b1, b2, b3], axis=-1)


def rotmat_to_sixd(rotmat: Float[Any, "... 3 3"], *, xp: ModuleType) -> Float[Any, "... 6"]:
    return xp.concatenate([rotmat[..., :, 0], rotmat[..., :, 1]], axis=-1)


def _normalize(vec: Float[Any, "... 3"], xp: ModuleType) -> Float[Any, "... 3"]:
    """Unit vector, or ``vec`` itself when its norm is below eps; a masked sqrt keeps JAX gradients finite at zero."""
    eps = common.safe_eps(vec.dtype, xp)
    norm_sq = xp.sum(vec * vec, axis=-1, keepdims=True)
    return vec / xp.sqrt(xp.where(norm_sq < eps * eps, xp.ones_like(norm_sq), norm_sq))
