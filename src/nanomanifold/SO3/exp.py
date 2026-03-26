from types import ModuleType
from typing import Any

from jaxtyping import Float

from .primitives.axis_angle import from_axis_angle
from .primitives.quaternion import QuaternionConvention, to_quat


def exp(
    tangent_vector: Float[Any, "... 3"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    """Compute the exponential map from so(3) tangent space to SO(3) manifold.

    The exponential map takes a tangent vector in the Lie algebra so(3)
    and returns the corresponding rotation quaternion. This is the inverse
    operation of log().

    The exponential map is mathematically equivalent to converting an axis-angle
    representation to its corresponding quaternion.

    Args:
        tangent_vector: Tangent vector in so(3) (axis-angle representation)
                       The magnitude is the rotation angle, direction is the rotation axis
        convention: Quaternion component order, either ``"wxyz"`` or ``"xyzw"``
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        Quaternion in the requested convention representing the rotation
    """
    return to_quat(from_axis_angle(tangent_vector, xp=xp), convention=convention, xp=xp)
