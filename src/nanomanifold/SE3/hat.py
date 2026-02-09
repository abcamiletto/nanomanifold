from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace
from nanomanifold.SO3 import hat as so3_hat


def hat(v: Float[Any, "... 6"], *, xp: ModuleType | None = None) -> Float[Any, "... 4 4"]:
    """Map 6-vector to 4x4 matrix in se(3) (hat operator).

    The input 6-vector is [omega, rho] where:
    - omega (first 3) is the angular velocity
    - rho (last 3) is the translational velocity

    The output 4x4 matrix has the form:
        [[SO3.hat(omega), rho],
         [0,  0,  0,       0 ]]

    Args:
        v: (..., 6) array representing tangent vectors in se(3)
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        (..., 4, 4) matrices in se(3)
    """
    if xp is None:
        xp = get_namespace(v)

    omega = v[..., :3]
    rho = v[..., 3:6]

    omega_hat = so3_hat(omega, xp=xp)  # (..., 3, 3)

    rho_col = rho[..., None]  # (..., 3, 1)
    top_block = xp.concatenate([omega_hat, rho_col], axis=-1)  # (..., 3, 4)

    zeros = xp.zeros_like(top_block[..., :1, :])  # (..., 1, 4)
    result = xp.concatenate([top_block, zeros], axis=-2)  # (..., 4, 4)

    return result
