from types import ModuleType
from typing import Any

from jaxtyping import Float

from .conversions.axis_angle import to_axis_angle


def log(q: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    """Compute the logarithmic map of a quaternion on the SO(3) manifold.

    The logarithmic map takes a rotation and returns the corresponding
    tangent vector (axis-angle representation) in the Lie algebra so(3).
    This is the inverse operation of exp().

    The logarithmic map is mathematically equivalent to converting a quaternion
    to its axis-angle representation.

    Args:
        q: Quaternion in [w, x, y, z] format representing a rotation
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        Tangent vector in so(3) (axis-angle representation)
        The magnitude is the rotation angle, direction is the rotation axis
    """
    return to_axis_angle(q, xp=xp)
