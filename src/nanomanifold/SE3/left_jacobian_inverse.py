r"""Inverse left Jacobian for :math:`\mathrm{SE}(3)` transformations."""

from typing import Any

from jaxtyping import Float

from ..common import get_namespace
from ..SO3.left_jacobian_inverse import left_jacobian_inverse as so3_left_jacobian_inverse
from ._jacobians import _jacobian_upper_right_block


def left_jacobian_inverse(tangent_vector: Float[Any, "... 6"]) -> Float[Any, "... 6 6"]:
    r"""Compute the inverse of the :math:`\mathrm{SE}(3)` left Jacobian."""

    xp = get_namespace(tangent_vector)

    omega = tangent_vector[..., :3]
    rho = tangent_vector[..., 3:6]

    J_inv = so3_left_jacobian_inverse(omega)
    Q = _jacobian_upper_right_block(rho, omega)

    correction = -xp.matmul(xp.matmul(J_inv, Q), J_inv)

    zeros = xp.zeros_like(J_inv)

    top = xp.concatenate([J_inv, zeros], axis=-1)
    bottom = xp.concatenate([correction, J_inv], axis=-1)

    return xp.concatenate([top, bottom], axis=-2)
