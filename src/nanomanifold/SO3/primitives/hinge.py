"""Hinge edges: hinge <-> axis_angle.

``axes`` are axis-angle generators. Unit axes make the angles physical rotation angles;
non-unit axes scale the generated rotation vector.
"""

from types import ModuleType
from typing import Any

from jaxtyping import Float


def hinge_to_axis_angle(angles: Float[Any, "... 1"], *, axes: Float[Any, "... 3"], xp: ModuleType) -> Float[Any, "... 3"]:
    assert angles.shape[-1:] == (1,), "Hinge angles must have shape (..., 1)."
    return angles * axes


def axis_angle_to_hinge(axis_angle: Float[Any, "... 3"], *, axes: Float[Any, "... 3"], xp: ModuleType) -> Float[Any, "... 1"]:
    """Least-squares coefficient of the rotation vector along ``axes``."""
    projected = xp.sum(axis_angle * axes, axis=-1, keepdims=True)
    axis_norm_sq = xp.sum(axes * axes, axis=-1, keepdims=True)
    return projected / axis_norm_sq
