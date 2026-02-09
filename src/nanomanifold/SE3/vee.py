from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace
from nanomanifold.SO3 import vee as so3_vee


def vee(M: Float[Any, "... 4 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    """Map 4x4 se(3) matrix to 6-vector (vee operator).

    Inverse of the hat operator. Extracts [omega, rho] from a 4x4 matrix:
    - omega from the top-left 3x3 skew-symmetric block via SO3.vee
    - rho from the top-right 3x1 column

    Args:
        M: (..., 4, 4) matrices in se(3)
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        (..., 6) tangent vectors in se(3) as [omega, rho]
    """
    if xp is None:
        xp = get_namespace(M)

    omega_hat = M[..., :3, :3]  # (..., 3, 3)
    rho = M[..., :3, 3]  # (..., 3)

    omega = so3_vee(omega_hat, xp=xp)  # (..., 3)

    return xp.concatenate([omega, rho], axis=-1)
