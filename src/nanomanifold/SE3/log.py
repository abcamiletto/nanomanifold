from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace
from nanomanifold.SO3 import log as so3_log
from nanomanifold.SO3.left_jacobian_inverse import (
    left_jacobian_inverse as so3_left_jacobian_inverse,
)

from .canonicalize import canonicalize


def log(se3: Float[Any, "... 7"]) -> Float[Any, "... 6"]:
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

    Returns:
        Tangent vector in se(3) as [ω, ρ] of shape (..., 6)
    """
    xp = get_namespace(se3)
    se3 = canonicalize(se3)

    q = se3[..., :4]
    t = se3[..., 4:7]

    omega = so3_log(q)
    V_inv = so3_left_jacobian_inverse(omega)
    rho = xp.matmul(V_inv, t[..., None])[..., 0]

    tangent = xp.concatenate([omega, rho], axis=-1)

    return tangent
