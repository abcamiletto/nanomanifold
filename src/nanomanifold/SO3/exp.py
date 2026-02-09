from types import ModuleType
from typing import Any

from jaxtyping import Float

from .conversions.axis_angle import from_axis_angle


def exp(tangent_vector: Float[Any, "... 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    """Compute the exponential map from so(3) tangent space to SO(3) manifold.

    The exponential map takes a tangent vector in the Lie algebra so(3)
    and returns the corresponding rotation quaternion. This is the inverse
    operation of log().

    The exponential map is mathematically equivalent to converting an axis-angle
    representation to its corresponding quaternion.

    Args:
        tangent_vector: Tangent vector in so(3) (axis-angle representation)
                       The magnitude is the rotation angle, direction is the rotation axis
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        Quaternion in [w, x, y, z] format representing the rotation
    """
    return from_axis_angle(tangent_vector, xp=xp)
