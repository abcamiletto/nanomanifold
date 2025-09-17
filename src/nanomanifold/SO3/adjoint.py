r"""Adjoint representation for :math:`\mathrm{SO}(3)` rotations."""

from typing import Any

from jaxtyping import Float

from .conversions.matrix import to_matrix


def adjoint(q: Float[Any, "... 4"]) -> Float[Any, "... 3 3"]:
    """Return the adjoint matrix of a unit quaternion.

    For :math:`\mathrm{SO}(3)` the adjoint representation coincides with the
    rotation matrix itself.

    Args:
        q: Quaternion of shape ``(..., 4)`` in ``[w, x, y, z]`` format.

    Returns:
        Array of shape ``(..., 3, 3)`` representing the adjoint action.
    """

    return to_matrix(q)
