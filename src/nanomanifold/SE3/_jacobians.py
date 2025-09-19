r"""Internal Jacobian helpers for :math:`\mathrm{SE}(3)`."""

from typing import Any

from jaxtyping import Float

from ..common import get_namespace
from ..SO3.hat import hat


def _jacobian_upper_right_block(rho: Float[Any, "... 3"], omega: Float[Any, "... 3"]) -> Float[Any, "... 3 3"]:
    r"""Compute the upper-right block of the :math:`\mathrm{SE}(3)` left Jacobian.

    This quantity captures how the translational component couples with rotation
    when moving on the :math:`\mathrm{SE}(3)` manifold. The implementation
    mirrors Sophus' closed-form expression and is shared between the Jacobian
    and its inverse.
    """

    xp = get_namespace(rho)

    dtype = rho.dtype
    rho_shape = rho.shape

    rho_arr = xp.asarray(rho, dtype=dtype)
    omega_arr = xp.asarray(omega, dtype=dtype)

    omega_sq_norm = xp.sum(omega_arr * omega_arr, axis=-1, keepdims=True)
    omega_norm = xp.sqrt(omega_sq_norm)

    Upsilon = hat(rho_arr)
    Omega = hat(omega_arr)

    eps = xp.finfo(dtype).eps
    small_angle_threshold = xp.asarray(max(1e-6, float(eps) * 10.0), dtype=dtype)
    small_angle_mask = omega_norm < small_angle_threshold

    Q_small = 0.5 * Upsilon

    half_norm = omega_norm / 2.0
    sin_half = xp.sin(half_norm)
    sin_theta = xp.sin(omega_norm)

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
        sin_theta / safe_norm,
    )

    A = 0.5 * sinc_half * sinc_half
    B = (1.0 - sinc) / safe_norm_sq

    inv_theta_sq = xp.reciprocal(safe_norm_sq)

    c1 = B
    c2 = (0.5 - A) * inv_theta_sq
    c3 = (1.5 * B - 0.5 * A) * inv_theta_sq

    c1 = xp.reshape(c1, c1.shape[:-1] + (1, 1))
    c2 = xp.reshape(c2, c2.shape[:-1] + (1, 1))
    c3 = xp.reshape(c3, c3.shape[:-1] + (1, 1))

    OmegaUpsilon = xp.matmul(Omega, Upsilon)
    UpsilonOmega = xp.matmul(Upsilon, Omega)
    OmegaUpsilonOmega = xp.matmul(OmegaUpsilon, Omega)

    theta_sq_matrix = xp.reshape(omega_sq_norm, omega_sq_norm.shape[:-1] + (1, 1))

    Q_large = (
        0.5 * Upsilon
        + c1 * (OmegaUpsilon + UpsilonOmega + OmegaUpsilonOmega)
        - c2 * (theta_sq_matrix * Upsilon + 2.0 * OmegaUpsilonOmega)
        + c3 * (xp.matmul(OmegaUpsilonOmega, Omega) + xp.matmul(Omega, OmegaUpsilonOmega))
    )

    mask = xp.reshape(small_angle_mask, small_angle_mask.shape[:-1] + (1, 1))
    Q = xp.where(mask, Q_small, Q_large)

    return xp.reshape(Q, rho_shape[:-1] + (3, 3))
