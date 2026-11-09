"""Axis-angle edges: axis_angle <-> canonical quaternion."""

import math
from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import common


def axis_angle_to_quat(axis_angle: Float[Any, "... 3"], *, xp: ModuleType) -> Float[Any, "... 4"]:
    angle = xp.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half_angle = angle / 2

    thresh = xp.asarray(math.sqrt(common.safe_eps(axis_angle.dtype, xp, scale=1.0)), dtype=axis_angle.dtype)
    small = angle < thresh
    safe_angle = xp.where(small, xp.ones_like(angle), angle)

    # Below the threshold sin(angle / 2) / angle ≈ 1 / 2, which keeps gradients finite at the identity.
    xyz_small = axis_angle / 2
    xyz_large = xp.sin(half_angle) * axis_angle / safe_angle
    xyz = xp.where(small, xyz_small, xyz_large)

    quat = xp.concatenate([xp.cos(half_angle), xyz], axis=-1)
    return xp.where(quat[..., 0:1] < 0, -quat, quat)


def quat_to_axis_angle(quat: Float[Any, "... 4"], *, xp: ModuleType) -> Float[Any, "... 3"]:
    w = quat[..., 0:1]
    xyz = quat[..., 1:4]
    norm_xyz = xp.linalg.norm(xyz, axis=-1, keepdims=True)

    thresh = xp.asarray(math.sqrt(common.safe_eps(quat.dtype, xp, scale=1.0)), dtype=quat.dtype)
    small = norm_xyz < thresh
    safe_norm = xp.where(small, xp.ones_like(norm_xyz), norm_xyz)

    angle = 2 * xp.arctan2(norm_xyz, xp.clip(w, 0.0, 1.0))
    axis_angle_small = 2 * xyz
    axis_angle_large = angle * xyz / safe_norm
    return xp.where(small, axis_angle_small, axis_angle_large)
