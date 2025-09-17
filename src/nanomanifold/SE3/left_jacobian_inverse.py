r"""Inverse left Jacobian for :math:`\mathrm{SE}(3)` transformations."""

from typing import Any

from jaxtyping import Float

from ..common import get_namespace
from ..SO3.left_jacobian_inverse import left_jacobian_inverse as so3_left_jacobian_inverse
from ._jacobians import _jacobian_upper_right_block


def left_jacobian_inverse(tangent_vector: Float[Any, "... 6"]) -> Float[Any, "... 6 6"]:
    r"""Compute the inverse of the :math:`\mathrm{SE}(3)` left Jacobian."""

    xp = get_namespace(tangent_vector)

    input_dtype = tangent_vector.dtype
    float16_dtype = getattr(xp, "float16", None)
    needs_upcast = float16_dtype is not None and input_dtype == float16_dtype
    calc_dtype = getattr(xp, "float32", input_dtype) if needs_upcast else input_dtype

    tangent_calc = xp.asarray(tangent_vector, dtype=calc_dtype)

    omega = tangent_calc[..., :3]
    rho = tangent_calc[..., 3:6]

    J_inv = so3_left_jacobian_inverse(omega)
    Q = _jacobian_upper_right_block(rho, omega)

    correction = -xp.matmul(xp.matmul(J_inv, Q), J_inv)

    zeros = xp.zeros_like(J_inv)

    top = xp.concatenate([J_inv, zeros], axis=-1)
    bottom = xp.concatenate([correction, J_inv], axis=-1)

    result = xp.concatenate([top, bottom], axis=-2)

    if needs_upcast:
        result = xp.asarray(result, dtype=input_dtype)

    return result
