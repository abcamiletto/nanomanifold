from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace
from nanomanifold.SO3 import exp as so3_exp
from nanomanifold.SO3 import hat


def exp(tangent_vector: Float[Any, "... 6"]) -> Float[Any, "... 7"]:
    """Compute the exponential map from se(3) tangent space to SE(3) manifold.

    The exponential map takes a tangent vector in the Lie algebra se(3)
    and returns the corresponding SE(3) transformation. This is the inverse
    operation of log().

    The se(3) exponential map takes a 6-vector [ω, ρ] where:
    - ω ∈ ℝ³ is the angular velocity (rotation part)
    - ρ ∈ ℝ³ is the translational velocity

    The formula involves:
    - R = exp_SO3(ω) for the rotation quaternion
    - t = V * ρ where V is the left Jacobian matrix

    Args:
        tangent_vector: Tangent vector in se(3) as [ω, ρ] of shape (..., 6)

    Returns:
        SE(3) transformation in [w, x, y, z, tx, ty, tz] format of shape (..., 7)
    """
    xp = get_namespace(tangent_vector)

    omega = tangent_vector[..., :3]
    rho = tangent_vector[..., 3:6]

    q = so3_exp(omega)
    omega_norm = xp.linalg.norm(omega, axis=-1, keepdims=True)

    eps = xp.finfo(omega.dtype).eps
    small_angle_threshold = xp.asarray(max(1e-6, float(eps)), dtype=omega.dtype)
    small_angle_mask = omega_norm < small_angle_threshold

    omega_cross = hat(omega)
    omega_cross_sq = xp.matmul(omega_cross, omega_cross)

    identity = xp.eye(3, dtype=omega.dtype)
    identity = xp.broadcast_to(identity, omega.shape[:-1] + (3, 3))

    V_small = identity + 0.5 * omega_cross + (1.0 / 12.0) * omega_cross_sq

    cos_norm = xp.cos(omega_norm)
    sin_norm = xp.sin(omega_norm)

    safe_norm = xp.where(small_angle_mask, xp.ones_like(omega_norm), omega_norm)
    safe_norm_sq = xp.where(small_angle_mask, xp.ones_like(omega_norm), omega_norm**2)
    safe_norm_cub = safe_norm_sq * safe_norm

    A = (1.0 - cos_norm) / safe_norm_sq
    B = (safe_norm - sin_norm) / safe_norm_cub
    A = xp.reshape(A, A.shape[:-1])[..., None, None]
    B = xp.reshape(B, B.shape[:-1])[..., None, None]

    V_large = identity + A * omega_cross + B * omega_cross_sq

    mask = xp.reshape(small_angle_mask, small_angle_mask.shape[:-1])[..., None, None]
    V = xp.where(mask, V_small, V_large)
    V = xp.reshape(V, omega.shape[:-1] + (3, 3))

    t = xp.matmul(V, rho[..., None])[..., 0]

    se3 = xp.concatenate([q, t], axis=-1)

    return se3
