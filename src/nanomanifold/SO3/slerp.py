from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace

from .canonicalize import canonicalize


def slerp(q1: Float[Any, "... 4"], q2: Float[Any, "... 4"], t: Float[Any, "... N"]) -> Float[Any, "... N 4"]:
"""Spherical linear interpolation between two quaternions representing SO(3).

The routine performs geodesic interpolation on the SO(3) manifold, taking the
shortest path between two rotations after canonicalizing the inputs to the same
hemisphere. The interpolation parameters ``t`` are typically chosen in the
closed interval ``[0, 1]`` where ``t = 0`` returns ``q1`` and ``t = 1`` returns
``q2``, but values outside that range are accepted and will extrapolate beyond
the arc connecting the endpoints.

Args:
    q1: Start quaternion in ``[w, x, y, z]`` format.
    q2: End quaternion in ``[w, x, y, z]`` format.
    t: Array of interpolation parameters whose last dimension ``N`` represents
        the number of interpolation points. For a single point use shape
        ``[..., 1]``.

Returns:
    Interpolated quaternions with shape ``[..., N, 4]`` where ``N`` is the last
    dimension of ``t``.
"""
    xp = get_namespace(q1)

    q1 = canonicalize(q1)
    q2 = canonicalize(q2)

    q1_expanded = xp.expand_dims(q1, axis=-2)
    q2_expanded = xp.expand_dims(q2, axis=-2)

    dot_product = xp.sum(q1_expanded * q2_expanded, axis=-1, keepdims=True)

    q2_corrected = xp.where(dot_product < 0, -q2_expanded, q2_expanded)
    dot_product = xp.where(dot_product < 0, -dot_product, dot_product)

    dot_product = xp.clip(dot_product, 0.0, 1.0)

    threshold = 0.9995

    t_expanded = xp.expand_dims(t, axis=-1)

    use_linear = dot_product > threshold

    linear_result = (1.0 - t_expanded) * q1_expanded + t_expanded * q2_corrected
    linear_norm = xp.sqrt(xp.sum(linear_result**2, axis=-1, keepdims=True))
    linear_result = linear_result / linear_norm

    omega = xp.acos(dot_product)
    sin_omega = xp.sin(omega)

    eps = xp.asarray(xp.finfo(q1.dtype).eps, dtype=q1.dtype)
    sin_omega_safe = xp.where(xp.abs(sin_omega) < eps, eps, sin_omega)

    weight1 = xp.sin((1.0 - t_expanded) * omega) / sin_omega_safe
    weight2 = xp.sin(t_expanded * omega) / sin_omega_safe

    spherical_result = weight1 * q1_expanded + weight2 * q2_corrected

    result = xp.where(use_linear, linear_result, spherical_result)

    return canonicalize(result)
