"""Axis-angle edges: axis_angle <-> canonical quaternion.

Norms are taken as a masked ``sqrt`` of the sum of squares: ``linalg.norm`` has no gradient at zero in JAX.
Below ``angle² < eps`` both directions switch to their first-order series, which are exact to machine precision there.
"""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import common

from .quaternion import positive_w


def axis_angle_to_quat(axis_angle: Float[Any, "... 3"], *, xp: ModuleType) -> Float[Any, "... 4"]:
    angle_sq = xp.sum(axis_angle * axis_angle, axis=-1, keepdims=True)
    small = angle_sq < common.safe_eps(axis_angle.dtype, xp, scale=1.0)
    angle = xp.sqrt(xp.where(small, xp.ones_like(angle_sq), angle_sq))

    # Below the threshold cos(angle / 2) ≈ 1 - angle² / 8 and sin(angle / 2) / angle ≈ 1 / 2.
    w = xp.where(small, 1 - angle_sq / 8, xp.cos(angle / 2))
    xyz = xp.where(small, axis_angle / 2, xp.sin(angle / 2) * axis_angle / angle)
    return positive_w(xp.concatenate([w, xyz], axis=-1), xp)


def quat_to_axis_angle(quat: Float[Any, "... 4"], *, xp: ModuleType) -> Float[Any, "... 3"]:
    w = quat[..., 0:1]
    xyz = quat[..., 1:4]
    norm_sq = xp.sum(xyz * xyz, axis=-1, keepdims=True)
    small = norm_sq < common.safe_eps(quat.dtype, xp, scale=1.0)
    norm = xp.sqrt(xp.where(small, xp.ones_like(norm_sq), norm_sq))

    angle = 2 * xp.arctan2(norm, xp.clip(w, 0.0, 1.0))
    return xp.where(small, 2 * xyz, angle * xyz / norm)
