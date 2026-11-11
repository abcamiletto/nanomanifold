"""Interpolation and averaging of SO(3) rotations."""

import math
from types import ModuleType
from typing import Any, Sequence

from jaxtyping import Float

from nanomanifold import common
from nanomanifold.common import get_namespace

from .primitives.quaternion import QuaternionConvention, _from_wxyz, from_quat, largest_eigenvector, normalize


def slerp(
    q1: Float[Any, "... 4"],
    q2: Float[Any, "... 4"],
    t: Float[Any, "... N"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... N 4"]:
    """Spherical linear interpolation along the shorter arc, at the ``N`` times in the last axis of ``t``.

    ``t = 0`` gives ``q1`` and ``t = 1`` gives ``q2``; values outside ``[0, 1]`` extrapolate. Returns ``(..., N, 4)``.
    """
    if xp is None:
        xp = get_namespace(q1)
    q1 = from_quat(q1, convention=convention, xp=xp)[..., None, :]
    q2 = from_quat(q2, convention=convention, xp=xp)[..., None, :]
    t = t[..., None]

    # Interpolate along the shorter arc, by negating the weight of q2 instead of q2 itself.
    dot = xp.sum(q1 * q2, axis=-1, keepdims=True)
    flip = dot < 0
    dot = xp.clip(xp.where(flip, -dot, dot), 0.0, 1.0)

    eps = common.safe_eps(dot.dtype, xp, scale=1.0)
    use_linear = dot > 1.0 - math.sqrt(eps)

    # Keep the acos argument away from exactly 1.0 so the unselected spherical
    # path does not introduce NaN gradients at the linear/spherical boundary.
    eps = xp.asarray(eps, dtype=dot.dtype)
    omega = xp.arccos(xp.minimum(dot, xp.ones_like(dot) - eps))
    sin_omega = xp.sin(omega)
    sin_omega = xp.where(xp.abs(sin_omega) < eps, eps, sin_omega)

    # Linear and spherical paths are both weighted sums: choose the (..., N, 1) weights, then combine once.
    weight1 = xp.where(use_linear, 1.0 - t, xp.sin((1.0 - t) * omega) / sin_omega)
    weight2 = xp.where(use_linear, t, xp.sin(t * omega) / sin_omega)
    weight2 = xp.where(flip, -weight2, weight2)

    # The linear branch needs renormalizing; the spherical one drifts off the unit sphere by rounding.
    return _from_wxyz(normalize(weight1 * q1 + weight2 * q2, xp), convention=convention, xp=xp)


def weighted_mean(
    qs: Sequence[Float[Any, "... 4"]],
    weights: Float[Any, "... N"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    """Weighted mean of ``N`` rotations ``(..., 4)``, with weights ``(..., N)`` normalized to sum to one.

    The mean is the top eigenvector of ``Σ wᵢ qᵢ qᵢᵀ`` (Markley et al., "Averaging Quaternions", 2007), which
    ignores the sign of each quaternion.
    """
    if xp is None:
        xp = get_namespace(qs[0])
    original_dtype = qs[0].dtype

    quats = xp.stack([from_quat(common.astype(q, original_dtype, xp), convention=convention, xp=xp) for q in qs], axis=-2)
    weights_array = common.astype(weights, original_dtype, xp)

    weights_normalized = weights_array / xp.sum(weights_array, axis=-1, keepdims=True)
    weighted_quats = weights_normalized[..., :, None] * quats
    M = xp.einsum("...nj,...nk->...jk", weighted_quats, quats)

    return _from_wxyz(largest_eigenvector(M, xp=xp), convention=convention, xp=xp)


def mean(
    qs: Sequence[Float[Any, "... 4"]], *, convention: QuaternionConvention = "wxyz", xp: ModuleType | None = None
) -> Float[Any, "... 4"]:
    """``weighted_mean`` with equal weights."""
    if len(qs) == 0:
        raise ValueError("Cannot compute mean of empty quaternion sequence")
    if xp is None:
        xp = get_namespace(qs[0])
    weights = xp.broadcast_to(xp.ones_like(qs[0][..., :1]), qs[0].shape[:-1] + (len(qs),))
    return weighted_mean(qs, weights, convention=convention, xp=xp)
