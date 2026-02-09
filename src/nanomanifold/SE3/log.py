from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import common
from nanomanifold.common import get_namespace
from nanomanifold.SO3 import hat
from nanomanifold.SO3 import log as so3_log

from .canonicalize import canonicalize


def log(se3: Float[Any, "... 7"], *, xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    """Compute the logarithmic map of SE(3) to its Lie algebra se(3).

    The logarithmic map takes an SE(3) transformation and returns the corresponding
    tangent vector in the Lie algebra se(3). This is the inverse operation of exp().

    The SE(3) logarithmic map computes a 6-vector [ω, ρ] where:
    - ω ∈ ℝ³ is the angular velocity (rotation part, same as SO(3) log)
    - ρ ∈ ℝ³ is the transformed translation part

    The formula involves:
    - ω = log_SO3(R) where R is the rotation quaternion
    - ρ = V^(-1) * t where V is the left Jacobian inverse and t is the translation

    Args:
        se3: SE(3) transformation in [w, x, y, z, tx, ty, tz] format of shape (..., 7)
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        Tangent vector in se(3) as [ω, ρ] of shape (..., 6)
    """
    if xp is None:
        xp = get_namespace(se3)
    se3 = canonicalize(se3, xp=xp)

    q = se3[..., :4]
    t = se3[..., 4:7]

    omega = so3_log(q, xp=xp)
    omega_norm = xp.linalg.norm(omega, axis=-1, keepdims=True)

    thresh = xp.asarray(common.small_angle_threshold(omega.dtype, xp), dtype=omega.dtype)
    small_angle_mask = omega_norm < thresh

    omega_cross = hat(omega, xp=xp)
    omega_cross_sq = xp.matmul(omega_cross, omega_cross)

    identity = xp.eye(3, dtype=omega.dtype)
    identity = xp.broadcast_to(identity, omega.shape[:-1] + (3, 3))

    V_inv_small = identity - 0.5 * omega_cross + (1.0 / 12.0) * omega_cross_sq

    half_norm = omega_norm / 2.0
    cos_half = xp.cos(half_norm)
    sin_half = xp.sin(half_norm)

    safe_sin_half = xp.where(small_angle_mask, xp.ones_like(sin_half), sin_half)
    cot_half = cos_half / safe_sin_half

    safe_norm = xp.where(small_angle_mask, xp.ones_like(omega_norm), omega_norm)
    safe_norm_sq = xp.where(small_angle_mask, xp.ones_like(omega_norm), omega_norm**2)
    B = (1.0 - 0.5 * safe_norm * cot_half) / safe_norm_sq
    B = xp.reshape(B, B.shape[:-1])[..., None, None]

    V_inv_large = identity - 0.5 * omega_cross + B * omega_cross_sq

    mask = xp.reshape(small_angle_mask, small_angle_mask.shape[:-1])[..., None, None]
    V_inv = xp.where(mask, V_inv_small, V_inv_large)
    V_inv = xp.reshape(V_inv, omega.shape[:-1] + (3, 3))

    rho = xp.matmul(V_inv, t[..., None])[..., 0]

    tangent = xp.concatenate([omega, rho], axis=-1)

    return tangent
