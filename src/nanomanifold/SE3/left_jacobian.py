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

    input_dtype = tangent_vector.dtype
    float16_dtype = getattr(xp, "float16", None)
    needs_upcast = float16_dtype is not None and input_dtype == float16_dtype
    calc_dtype = getattr(xp, "float32", input_dtype) if needs_upcast else input_dtype

    tangent_calc = xp.asarray(tangent_vector, dtype=calc_dtype)

    omega = tangent_calc[..., :3]
    rho = tangent_calc[..., 3:6]

    J = so3_left_jacobian(omega)
    Q = _jacobian_upper_right_block(rho, omega)

    zeros = xp.zeros_like(J)

    top = xp.concatenate([J, zeros], axis=-1)
    bottom = xp.concatenate([Q, J], axis=-1)

    result = xp.concatenate([top, bottom], axis=-2)

    if needs_upcast:
        result = xp.asarray(result, dtype=input_dtype)

    return result
