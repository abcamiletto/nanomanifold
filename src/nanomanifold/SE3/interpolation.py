"""Interpolation and averaging of SE(3) transforms: the rotation as in SO(3), the translation linearly."""

from types import ModuleType
from typing import Any, Sequence

from jaxtyping import Float

from nanomanifold import SO3, common
from nanomanifold.common import get_namespace
from nanomanifold.SO3.primitives.quaternion import QuaternionConvention


def slerp(
    T1: Float[Any, "... 7"],
    T2: Float[Any, "... 7"],
    t: Float[Any, "... N"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... N 7"]:
    """Interpolate at the ``N`` times in the last axis of ``t``: ``SO3.slerp`` for rotations, linear for translations.

    ``t = 0`` gives ``T1`` and ``t = 1`` gives ``T2``. Returns ``(..., N, 7)``.
    """
    if xp is None:
        xp = get_namespace(T1)
    q = SO3.slerp(T1[..., :4], T2[..., :4], t, convention=convention, xp=xp)  # (..., N, 4)
    t = t[..., None]
    translation = (1.0 - t) * T1[..., None, 4:7] + t * T2[..., None, 4:7]  # (..., N, 3)
    return xp.concatenate([q, translation], axis=-1)


def weighted_mean(
    Ts: Sequence[Float[Any, "... 7"]],
    weights: Float[Any, "... N"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 7"]:
    """Weighted mean of ``N`` transforms ``(..., 7)``, with weights ``(..., N)`` normalized to sum to one.

    The rotation is ``SO3.weighted_mean`` of the rotations, the translation the weighted average of the translations.
    """
    if xp is None:
        xp = get_namespace(Ts[0])

    q_mean = SO3.weighted_mean([T[..., :4] for T in Ts], weights, convention=convention, xp=xp)

    weights = common.astype(weights, Ts[0].dtype, xp)
    weights = weights / xp.sum(weights, axis=-1, keepdims=True)
    t_stack = xp.stack([T[..., 4:7] for T in Ts], axis=-2)  # (..., N, 3)
    t_mean = xp.sum(weights[..., :, None] * t_stack, axis=-2)

    return xp.concatenate([q_mean, t_mean], axis=-1)


def mean(
    Ts: Sequence[Float[Any, "... 7"]], *, convention: QuaternionConvention = "wxyz", xp: ModuleType | None = None
) -> Float[Any, "... 7"]:
    """``weighted_mean`` with equal weights."""
    if len(Ts) == 0:
        raise ValueError("Cannot compute mean of empty transform sequence")
    if xp is None:
        xp = get_namespace(Ts[0])
    weights = xp.broadcast_to(xp.ones_like(Ts[0][..., :1]), Ts[0].shape[:-1] + (len(Ts),))
    return weighted_mean(Ts, weights, convention=convention, xp=xp)
