r"""Left Jacobian for :math:`\mathrm{SE}(3)` transformations."""

from typing import Any

from jaxtyping import Float

from ..common import get_namespace
from ..SO3.left_jacobian import left_jacobian as so3_left_jacobian
from ._jacobians import _jacobian_upper_right_block


def left_jacobian(tangent_vector: Float[Any, "... 6"]) -> Float[Any, "... 6 6"]:
    r"""Compute the left Jacobian of :math:`\mathrm{SE}(3)`.

    Args:
        tangent_vector: Array of shape ``(..., 6)`` containing the rotation part
            ``omega`` followed by the translation part ``rho``.

    Returns:
        Array of shape ``(..., 6, 6)`` representing the left Jacobian matrix.
    """

    xp = get_namespace(tangent_vector)

    omega = tangent_vector[..., :3]
    rho = tangent_vector[..., 3:6]

    J = so3_left_jacobian(omega)
    Q = _jacobian_upper_right_block(rho, omega)

    zeros = xp.zeros_like(J)

    top = xp.concatenate([J, zeros], axis=-1)
    bottom = xp.concatenate([Q, J], axis=-1)

    return xp.concatenate([top, bottom], axis=-2)
