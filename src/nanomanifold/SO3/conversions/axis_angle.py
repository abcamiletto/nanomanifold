from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace
from nanomanifold.SO3.canonicalize import canonicalize


def to_axis_angle(q: Float[Any, "... 4"]) -> Float[Any, "... 3"]:
    xp = get_namespace(q)
    q_canonical = canonicalize(q)

    w = q_canonical[..., 0:1]
    xyz = q_canonical[..., 1:4]

    norm_xyz = xp.linalg.norm(xyz, axis=-1, keepdims=True)

    eps = xp.finfo(q.dtype).eps
    small_angle_threshold = xp.asarray(max(1e-6, float(eps) * 10.0), dtype=q.dtype)
    small_angle_mask = norm_xyz < small_angle_threshold

    axis_angle_small = 2.0 * xyz

    w_clipped = xp.clip(w, 0.0, 1.0)
    angle = 2 * xp.atan2(norm_xyz, w_clipped)

    safe_norm = xp.where(norm_xyz < small_angle_threshold, xp.ones_like(norm_xyz), norm_xyz)
    axis = xyz / safe_norm

    axis_angle_large = angle * axis

    axis_angle = xp.where(small_angle_mask, axis_angle_small, axis_angle_large)

    return axis_angle


def from_axis_angle(axis_angle: Float[Any, "... 3"]) -> Float[Any, "... 4"]:
    xp = get_namespace(axis_angle)

    angle = xp.linalg.norm(axis_angle, axis=-1)

    eps = xp.finfo(axis_angle.dtype).eps
    small_angle_threshold = xp.asarray(max(1e-6, float(eps) * 10.0), dtype=axis_angle.dtype)
    small_angle_mask = angle < small_angle_threshold
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

    q = xp.concat([w, xyz], axis=-1)

    mask = q[..., 0:1] < 0
    q = xp.where(mask, -q, q)

    return q
