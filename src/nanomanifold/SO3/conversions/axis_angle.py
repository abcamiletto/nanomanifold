from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import common
from nanomanifold.common import get_namespace

from .quaternion import canonicalize


def to_axis_angle(q: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    if xp is None:
        xp = get_namespace(q)
    q_canonical = canonicalize(q, xp=xp)

    w = q_canonical[..., 0:1]
    xyz = q_canonical[..., 1:4]

    norm_xyz = xp.linalg.norm(xyz, axis=-1, keepdims=True)

    thresh = xp.asarray(common.small_angle_threshold(q.dtype, xp), dtype=q.dtype)
    small_angle_mask = norm_xyz < thresh

    axis_angle_small = 2.0 * xyz

    w_clipped = xp.clip(w, 0.0, 1.0)
    angle = 2 * xp.atan2(norm_xyz, w_clipped)

    safe_norm = xp.where(norm_xyz < thresh, xp.ones_like(norm_xyz), norm_xyz)
    axis = xyz / safe_norm

    axis_angle_large = angle * axis

    axis_angle = xp.where(small_angle_mask, axis_angle_small, axis_angle_large)

    return axis_angle


def from_axis_angle(axis_angle: Float[Any, "... 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    if xp is None:
        xp = get_namespace(axis_angle)

    angle = xp.linalg.norm(axis_angle, axis=-1)

    thresh = xp.asarray(common.small_angle_threshold(axis_angle.dtype, xp), dtype=axis_angle.dtype)
    small_angle_mask = angle < thresh
    safe_angle = xp.where(small_angle_mask, xp.ones_like(angle), angle)
    axis = axis_angle / safe_angle[..., None]

    half_angle = angle / 2

    # For small angles, use Taylor series: cos(x) ≈ 1 - x²/2, sin(x) ≈ x
    cos_half = xp.cos(half_angle)
    sin_half = xp.sin(half_angle)

    w = cos_half[..., None]

    xyz_normal = sin_half[..., None] * axis
    xyz_small = axis_angle / 2.0

    xyz = xp.where(small_angle_mask[..., None], xyz_small, xyz_normal)

    q = xp.concatenate([w, xyz], axis=-1)

    mask = q[..., 0:1] < 0
    q = xp.where(mask, -q, q)

    return q
