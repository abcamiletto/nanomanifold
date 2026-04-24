"""Helpers for one-axis SO(3) hinge rotations."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import common

from .primitives import axis_angle as axis_angle_primitives
from .primitives.quaternion import QuaternionConvention


def from_hinge(
    angles: Float[Any, "... 1"],
    axes: Float[Any, "... 3"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    """Create rotations constrained to a single hinge axis.

    ``axes`` are interpreted as axis-angle generators. Unit axes make ``angles``
    physical rotation angles; non-unit axes scale the generated rotation vector.
    """
    if xp is None:
        xp = common.get_namespace(angles)

    assert angles.shape[-1:] == (1,), "Hinge angles must have shape (..., 1)."
    axis_angle = angles * axes
    return axis_angle_primitives.from_axis_angle(axis_angle, convention=convention, xp=xp)


def to_hinge(
    q: Float[Any, "... 4"],
    axes: Float[Any, "... 3"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 1"]:
    """Project an SO(3) rotation onto a single hinge axis.

    The result is the least-squares coefficient of the rotation vector along
    ``axes``. For unit axes this is the signed principal rotation angle.
    """
    if xp is None:
        xp = common.get_namespace(q)

    axis_angle = axis_angle_primitives.to_axis_angle(q, convention=convention, xp=xp)
    projected = xp.sum(axis_angle * axes, axis=-1, keepdims=True)
    axis_norm_sq = xp.sum(axes * axes, axis=-1, keepdims=True)
    return projected / axis_norm_sq


__all__ = ["from_hinge", "to_hinge"]
