r"""Inverse of the left Jacobian for :math:`\mathrm{SO}(3)` rotations."""

from typing import Any

from jaxtyping import Float

from ..common import get_namespace
from .hat import hat


def left_jacobian_inverse(omega: Float[Any, "... 3"]) -> Float[Any, "... 3 3"]:
    r"""Compute the inverse of the :math:`\mathrm{SO}(3)` left Jacobian.

    Args:
        omega: Rotation vector of shape ``(..., 3)`` in axis-angle form.

    Returns:
        Array of shape ``(..., 3, 3)`` representing the inverse left Jacobian.
    """

    xp = get_namespace(omega)

    omega_norm = xp.linalg.norm(omega, axis=-1, keepdims=True)
    omega_cross = hat(omega)
    omega_cross_sq = xp.matmul(omega_cross, omega_cross)

    identity = xp.eye(3, dtype=omega.dtype)
    identity = xp.broadcast_to(identity, omega.shape[:-1] + (3, 3))

    eps = xp.finfo(omega.dtype).eps
    small_angle_threshold = xp.asarray(max(1e-6, float(eps) * 10.0), dtype=omega.dtype)
    small_angle_mask = omega_norm < small_angle_threshold

    J_inv_small = identity - 0.5 * omega_cross + (1.0 / 12.0) * omega_cross_sq

    half_norm = omega_norm / 2.0
    cos_half = xp.cos(half_norm)
    sin_half = xp.sin(half_norm)

    safe_sin_half = xp.where(small_angle_mask, xp.ones_like(sin_half), sin_half)
    cot_half = cos_half / safe_sin_half

    safe_norm = xp.where(small_angle_mask, xp.ones_like(omega_norm), omega_norm)
    safe_norm_sq = xp.where(small_angle_mask, xp.ones_like(omega_norm), omega_norm**2)

    B = (1.0 - 0.5 * safe_norm * cot_half) / safe_norm_sq
    B = xp.reshape(B, B.shape[:-1] + (1, 1))

    J_inv_large = identity - 0.5 * omega_cross + B * omega_cross_sq

    mask = xp.reshape(small_angle_mask, small_angle_mask.shape[:-1] + (1, 1))
    J_inv = xp.where(mask, J_inv_small, J_inv_large)

    return xp.reshape(J_inv, omega.shape[:-1] + (3, 3))
