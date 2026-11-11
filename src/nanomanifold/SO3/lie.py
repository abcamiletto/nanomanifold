"""Lie-algebra maps of SO(3): ``exp``/``log`` between rotations and so(3), ``hat``/``vee`` between vectors and matrices."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace

from .convert import from_axis_angle, to_axis_angle
from .primitives.quaternion import QuaternionConvention


def exp(w: Float[Any, "... 3"], *, convention: QuaternionConvention = "wxyz", xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    """Map so(3) tangent vectors ``(..., 3)`` to quaternions ``(..., 4)``; the same as ``from_axis_angle``."""
    return from_axis_angle(w, convention=convention, xp=xp)


def log(q: Float[Any, "... 4"], *, convention: QuaternionConvention = "wxyz", xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    """Map quaternions ``(..., 4)`` to so(3) tangent vectors ``(..., 3)`` of norm ``<= π``; the same as ``to_axis_angle``."""
    return to_axis_angle(q, convention=convention, xp=xp)


def hat(w: Float[Any, "... 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 3 3"]:
    """Skew-symmetric matrices ``(..., 3, 3)`` of vectors ``(..., 3)``, so that ``hat(w) @ p == cross(w, p)``."""
    if xp is None:
        xp = get_namespace(w)
    w1, w2, w3 = w[..., 0], w[..., 1], w[..., 2]
    zero = xp.zeros_like(w1)
    rows = [[zero, -w3, w2], [w3, zero, -w1], [-w2, w1, zero]]
    return xp.stack([xp.stack(row, axis=-1) for row in rows], axis=-2)


def vee(W: Float[Any, "... 3 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    """Inverse of ``hat``: the vectors ``(..., 3)`` of skew-symmetric matrices ``(..., 3, 3)``."""
    if xp is None:
        xp = get_namespace(W)
    return xp.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], axis=-1)
