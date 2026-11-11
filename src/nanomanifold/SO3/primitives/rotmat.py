"""Rotation-matrix edges: rotmat <-> canonical quaternion."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import common

from . import quaternion


def rotmat_to_quat(rotmat: Float[Any, "... 3 3"], *, xp: ModuleType) -> Float[Any, "... 4"]:
    """Shepperd's method: use the best-conditioned of four candidate solutions."""
    trace = rotmat[..., 0, 0] + rotmat[..., 1, 1] + rotmat[..., 2, 2]

    zero = xp.zeros_like(trace)
    one = xp.ones_like(trace)
    eps = xp.asarray(common.safe_eps(rotmat.dtype, xp), dtype=rotmat.dtype)
    two = one + one
    quarter = one / (two + two)

    s1 = xp.sqrt(xp.maximum(zero + eps, trace + one)) * two
    s1_safe = xp.where(s1 < eps, eps, s1)
    w1 = quarter * s1
    x1 = (rotmat[..., 2, 1] - rotmat[..., 1, 2]) / s1_safe
    y1 = (rotmat[..., 0, 2] - rotmat[..., 2, 0]) / s1_safe
    z1 = (rotmat[..., 1, 0] - rotmat[..., 0, 1]) / s1_safe

    s2 = xp.sqrt(xp.maximum(zero + eps, one + rotmat[..., 0, 0] - rotmat[..., 1, 1] - rotmat[..., 2, 2])) * two
    s2_safe = xp.where(s2 < eps, eps, s2)
    w2 = (rotmat[..., 2, 1] - rotmat[..., 1, 2]) / s2_safe
    x2 = quarter * s2
    y2 = (rotmat[..., 0, 1] + rotmat[..., 1, 0]) / s2_safe
    z2 = (rotmat[..., 0, 2] + rotmat[..., 2, 0]) / s2_safe

    s3 = xp.sqrt(xp.maximum(zero + eps, one + rotmat[..., 1, 1] - rotmat[..., 0, 0] - rotmat[..., 2, 2])) * two
    s3_safe = xp.where(s3 < eps, eps, s3)
    w3 = (rotmat[..., 0, 2] - rotmat[..., 2, 0]) / s3_safe
    x3 = (rotmat[..., 0, 1] + rotmat[..., 1, 0]) / s3_safe
    y3 = quarter * s3
    z3 = (rotmat[..., 1, 2] + rotmat[..., 2, 1]) / s3_safe

    s4 = xp.sqrt(xp.maximum(zero + eps, one + rotmat[..., 2, 2] - rotmat[..., 0, 0] - rotmat[..., 1, 1])) * two
    s4_safe = xp.where(s4 < eps, eps, s4)
    w4 = (rotmat[..., 1, 0] - rotmat[..., 0, 1]) / s4_safe
    x4 = (rotmat[..., 0, 2] + rotmat[..., 2, 0]) / s4_safe
    y4 = (rotmat[..., 1, 2] + rotmat[..., 2, 1]) / s4_safe
    z4 = quarter * s4

    cond1 = trace > 0
    cond2 = (rotmat[..., 0, 0] > rotmat[..., 1, 1]) & (rotmat[..., 0, 0] > rotmat[..., 2, 2])
    cond3 = rotmat[..., 1, 1] > rotmat[..., 2, 2]

    w = xp.where(cond1, w1, xp.where(cond2, w2, xp.where(cond3, w3, w4)))
    x = xp.where(cond1, x1, xp.where(cond2, x2, xp.where(cond3, x3, x4)))
    y = xp.where(cond1, y1, xp.where(cond2, y2, xp.where(cond3, y3, y4)))
    z = xp.where(cond1, z1, xp.where(cond2, z2, xp.where(cond3, z3, z4)))

    # Unit only for an exactly orthonormal input, so normalize rather than just flip the sign.
    return quaternion.normalize(xp.stack([w, x, y, z], axis=-1), xp)


def quat_to_rotmat(q: Float[Any, "... 4"], *, xp: ModuleType) -> Float[Any, "... 3 3"]:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    one = xp.ones_like(w)
    two = one + one

    return xp.stack(
        [
            xp.stack([one - two * (y * y + z * z), two * (x * y - w * z), two * (x * z + w * y)], axis=-1),
            xp.stack([two * (x * y + w * z), one - two * (x * x + z * z), two * (y * z - w * x)], axis=-1),
            xp.stack([two * (x * z - w * y), two * (y * z + w * x), one - two * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )
