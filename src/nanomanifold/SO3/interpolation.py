import math
from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import common
from nanomanifold.common import get_namespace

from .primitives.quaternion import QuaternionConvention, canonicalize, from_quat, to_quat


def slerp(
    q1: Float[Any, "... 4"],
    q2: Float[Any, "... 4"],
    t: Float[Any, "... N"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... N 4"]:
    """Spherical linear interpolation between two quaternions representing SO(3).

    The routine performs geodesic interpolation on the SO(3) manifold, taking the
    shortest path between two rotations after canonicalizing the inputs to the same
    hemisphere. The interpolation parameters ``t`` are typically chosen in the
    closed interval ``[0, 1]`` where ``t = 0`` returns ``q1`` and ``t = 1`` returns
    ``q2``, but values outside that range are accepted and will extrapolate beyond
    the arc connecting the endpoints.

    Args:
        q1: Start quaternion in the given convention.
        q2: End quaternion in the given convention.
        t: Array of interpolation parameters whose last dimension ``N`` represents
            the number of interpolation points. For a single point use shape
            ``[..., 1]``.
        convention: Quaternion component order, either ``"wxyz"`` or ``"xyzw"``
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        Interpolated quaternions with shape ``[..., N, 4]`` where ``N`` is the last
        dimension of ``t``.
    """
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace(q1)

    q1 = canonicalize(from_quat(q1, convention=convention, xp=xp), xp=xp)
    q2 = canonicalize(from_quat(q2, convention=convention, xp=xp), xp=xp)

    q1_expanded = q1[..., None, :]
    q2_expanded = q2[..., None, :]

    dot_product = xp.sum(q1_expanded * q2_expanded, axis=-1, keepdims=True)

    q2_corrected = xp.where(dot_product < 0, -q2_expanded, q2_expanded)
    dot_product = xp.where(dot_product < 0, -dot_product, dot_product)

    dot_product = xp.clip(dot_product, 0.0, 1.0)

    threshold = 1.0 - math.sqrt(common.safe_eps(dot_product.dtype, xp, scale=1.0))

    t_expanded = t[..., None]

    use_linear = dot_product > threshold

    linear_result = (1.0 - t_expanded) * q1_expanded + t_expanded * q2_corrected
    linear_norm = xp.sqrt(xp.sum(linear_result**2, axis=-1, keepdims=True))
    linear_result = linear_result / linear_norm

    # Keep the acos argument away from exactly 1.0 so the unselected spherical
    # path does not introduce NaN gradients at the linear/spherical boundary.
    one = xp.ones_like(dot_product)
    eps = xp.asarray(common.safe_eps(dot_product.dtype, xp, scale=1.0), dtype=dot_product.dtype)
    dot_for_trig = xp.minimum(dot_product, one - eps)
    omega = xp.arccos(dot_for_trig)
    sin_omega = xp.sin(omega)

    sin_omega_safe = xp.where(xp.abs(sin_omega) < eps, eps, sin_omega)

    weight1 = xp.sin((1.0 - t_expanded) * omega) / sin_omega_safe
    weight2 = xp.sin(t_expanded * omega) / sin_omega_safe

    spherical_result = weight1 * q1_expanded + weight2 * q2_corrected

    result = xp.where(use_linear, linear_result, spherical_result)

    return to_quat(canonicalize(result, xp=xp), convention=convention, xp=xp)
