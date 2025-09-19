from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace
from nanomanifold.SO3 import exp as so3_exp
from nanomanifold.SO3.left_jacobian import left_jacobian as so3_left_jacobian


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
    V = so3_left_jacobian(omega)
    t = xp.matmul(V, rho[..., None])[..., 0]

    se3 = xp.concatenate([q, t], axis=-1)

    return se3
