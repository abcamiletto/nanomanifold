from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import common
from nanomanifold.common import get_namespace

from .convert import RotationSourceRep, convert


def distance(
    q1: Float[Any, "..."],
    q2: Float[Any, "..."],
    *,
    rotation_type: RotationSourceRep = "quat",
    convention: str = "wxyz",
    rot_kwargs: dict[str, Any] = {},
    xp: ModuleType | None = None,
) -> Float[Any, "..."]:
    """Compute the angular distance between two SO(3) rotations.

    The angular distance is the smallest angle needed to rotate from one orientation
    to another, measured in radians. This is equivalent to the geodesic distance
    on the SO(3) manifold.

    Args:
        q1: First rotation
        q2: Second rotation
        rotation_type: Representation shared by q1 and q2
        convention: Convention used when ``rotation_type`` is ``"euler"`` or ``"quat"``
        rot_kwargs: Representation-specific keyword arguments passed to ``SO3.convert``
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        Angular distance in radians, in range [0, π]
    """
    if xp is None:
        xp = get_namespace(q1)
    q1 = convert(q1, src=rotation_type, dst="quat", src_kwargs={"convention": convention, **rot_kwargs}, xp=xp)
    q2 = convert(q2, src=rotation_type, dst="quat", src_kwargs={"convention": convention, **rot_kwargs}, xp=xp)

    eps = common.safe_eps(q1.dtype, xp)
    eps_arr = xp.asarray(eps, dtype=q1.dtype)
    eps_sq = eps_arr * eps_arr

    norm1 = xp.linalg.norm(q1, axis=-1, keepdims=True)
    norm2 = xp.linalg.norm(q2, axis=-1, keepdims=True)
    q1_unit = q1 / xp.maximum(norm1, eps_arr)
    q2_unit = q2 / xp.maximum(norm2, eps_arr)

    w1 = q1_unit[..., :1]
    v1 = q1_unit[..., 1:]
    w2 = q2_unit[..., :1]
    v2 = q2_unit[..., 1:]

    cross = xp.stack(
        [
            v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
            v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
            v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0],
        ],
        axis=-1,
    )

    vec = w1 * v2 - w2 * v1 - cross
    vec_sq = xp.sum(vec * vec, axis=-1)
    vec_norm = xp.sqrt(xp.maximum(vec_sq, eps_sq))
    w = w1 * w2 + xp.sum(v1 * v2, axis=-1, keepdims=True)
    w_abs = xp.abs(w[..., 0])
    two = xp.ones_like(vec_norm) + xp.ones_like(vec_norm)
    angle = two * xp.arctan2(vec_norm, w_abs)

    return xp.where(vec_sq <= eps_sq, xp.zeros_like(angle), angle)
