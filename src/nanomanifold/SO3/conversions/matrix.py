"""Matrix conversions for SO(3) rotations."""

from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace

from ..canonicalize import canonicalize


def to_matrix(q: Float[Any, "... 4"]) -> Float[Any, "... 3 3"]:
    xp = get_namespace(q)
    q = canonicalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = xp.stack(
        [
            xp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            xp.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            xp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )

    return R


def from_matrix(R: Float[Any, "... 3 3"]) -> Float[Any, "... 4"]:
    xp = get_namespace(R)

    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    zero = trace * 0
    one = zero + 1
    eps = one * 1e-10
    quarter = one * 0.25
    two = one * 2

    s1 = xp.sqrt(xp.maximum(zero, trace + one)) * two  # s = 4 * w
    s1_safe = xp.where(s1 < eps, eps, s1)  # Avoid division by zero
    w1 = quarter * s1
    x1 = (R[..., 2, 1] - R[..., 1, 2]) / s1_safe
    y1 = (R[..., 0, 2] - R[..., 2, 0]) / s1_safe
    z1 = (R[..., 1, 0] - R[..., 0, 1]) / s1_safe

    s2 = xp.sqrt(xp.maximum(zero, one + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2])) * two  # s = 4 * x
    s2_safe = xp.where(s2 < eps, eps, s2)  # Avoid division by zero
    w2 = (R[..., 2, 1] - R[..., 1, 2]) / s2_safe
    x2 = quarter * s2
    y2 = (R[..., 0, 1] + R[..., 1, 0]) / s2_safe
    z2 = (R[..., 0, 2] + R[..., 2, 0]) / s2_safe

    s3 = xp.sqrt(xp.maximum(zero, one + R[..., 1, 1] - R[..., 0, 0] - R[..., 2, 2])) * two  # s = 4 * y
    s3_safe = xp.where(s3 < eps, eps, s3)  # Avoid division by zero
    w3 = (R[..., 0, 2] - R[..., 2, 0]) / s3_safe
    x3 = (R[..., 0, 1] + R[..., 1, 0]) / s3_safe
    y3 = quarter * s3
    z3 = (R[..., 1, 2] + R[..., 2, 1]) / s3_safe

    s4 = xp.sqrt(xp.maximum(zero, one + R[..., 2, 2] - R[..., 0, 0] - R[..., 1, 1])) * two  # s = 4 * z
    s4_safe = xp.where(s4 < eps, eps, s4)  # Avoid division by zero
    w4 = (R[..., 1, 0] - R[..., 0, 1]) / s4_safe
    x4 = (R[..., 0, 2] + R[..., 2, 0]) / s4_safe
    y4 = (R[..., 1, 2] + R[..., 2, 1]) / s4_safe
    z4 = quarter * s4

    cond1 = trace > 0
    cond2 = (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
    cond3 = R[..., 1, 1] > R[..., 2, 2]

    w = xp.where(cond1, w1, xp.where(cond2, w2, xp.where(cond3, w3, w4)))
    x = xp.where(cond1, x1, xp.where(cond2, x2, xp.where(cond3, x3, x4)))
    y = xp.where(cond1, y1, xp.where(cond2, y2, xp.where(cond3, y3, y4)))
    z = xp.where(cond1, z1, xp.where(cond2, z2, xp.where(cond3, z3, z4)))

    q = xp.stack([w, x, y, z], axis=-1)

    return canonicalize(q)
