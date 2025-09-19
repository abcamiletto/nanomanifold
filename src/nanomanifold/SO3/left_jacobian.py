r"""Left Jacobian for :math:`\mathrm{SO}(3)` rotations."""

from typing import Any

from jaxtyping import Float

from ..common import get_namespace
from .hat import hat


def left_jacobian(omega: Float[Any, "... 3"]) -> Float[Any, "... 3 3"]:
    r"""Compute the left Jacobian of :math:`\mathrm{SO}(3)`.

    The left Jacobian maps perturbations in the Lie algebra :math:`\mathfrak{so}(3)`
    to perturbations on the manifold. It is defined as

    .. math::

        J_l(\boldsymbol{\omega}) = \sum_{n=0}^{\infty} \frac{1}{(n+1)!} \operatorname{ad}_{\boldsymbol{\omega}}^n

    but can be written in closed form using trigonometric functions. The same
    Jacobian appears in the translational part of the :math:`\mathrm{SE}(3)`
    exponential map, making it a key building block for robotics algorithms.

    Args:
        omega: Rotation vector of shape ``(..., 3)`` in axis-angle form.

    Returns:
        Array of shape ``(..., 3, 3)`` representing the left Jacobian.
    """

    xp = get_namespace(omega)

    dtype = omega.dtype
    omega_shape = omega.shape

    omega_norm = xp.linalg.norm(omega, axis=-1, keepdims=True)
    omega_cross = hat(omega)
    omega_cross_sq = xp.matmul(omega_cross, omega_cross)

    identity = xp.eye(3, dtype=dtype)
    identity = xp.broadcast_to(identity, omega_shape[:-1] + (3, 3))

    eps = xp.finfo(dtype).eps
    small_angle_threshold = xp.asarray(max(1e-6, float(eps) * 10.0), dtype=dtype)
    small_angle_mask = omega_norm < small_angle_threshold

    J_small = identity + 0.5 * omega_cross + (1.0 / 12.0) * omega_cross_sq

    half_norm = omega_norm / 2.0
    sin_half = xp.sin(half_norm)
    sin_norm = xp.sin(omega_norm)

    safe_norm = xp.where(small_angle_mask, xp.ones_like(omega_norm), omega_norm)
    safe_norm_sq = safe_norm * safe_norm
    safe_half_norm = xp.where(small_angle_mask, xp.ones_like(half_norm), half_norm)

    sinc_half = xp.where(
        small_angle_mask,
        xp.ones_like(half_norm),
        sin_half / safe_half_norm,
    )
    sinc = xp.where(
        small_angle_mask,
        xp.ones_like(omega_norm),
        sin_norm / safe_norm,
    )

    A = 0.5 * sinc_half * sinc_half
    B = (1.0 - sinc) / safe_norm_sq

    A = xp.reshape(A, A.shape[:-1] + (1, 1))
    B = xp.reshape(B, B.shape[:-1] + (1, 1))

    J_large = identity + A * omega_cross + B * omega_cross_sq

    mask = xp.reshape(small_angle_mask, small_angle_mask.shape[:-1] + (1, 1))
    J = xp.where(mask, J_small, J_large)

    return xp.reshape(J, omega_shape[:-1] + (3, 3))
