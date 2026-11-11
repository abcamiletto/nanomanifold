"""Euler edges: euler <-> canonical quaternion.

Uppercase conventions are intrinsic (rotations about the moving axes), lowercase are extrinsic
(rotations about the fixed axes).
"""

import math
from types import ModuleType
from typing import Any, Literal, get_args

from jaxtyping import Float

from nanomanifold import common

from . import quaternion

EulerConvention = Literal[
    "xyz", "xzy", "yxz", "yzx", "zxy", "zyx",
    "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX",
    "xyx", "xzx", "yxy", "yzy", "zxz", "zyz",
    "XYX", "XZX", "YXY", "YZY", "ZXZ", "ZYZ",
]  # fmt: skip

EULER_CONVENTIONS = get_args(EulerConvention)


def _check(convention: str) -> None:
    assert convention in EULER_CONVENTIONS, f"Invalid Euler convention {convention!r}: expected three axes such as 'ZYX' or 'xyz'."


def euler_to_quat(euler: Float[Any, "... 3"], *, convention: EulerConvention = "ZYX", xp: ModuleType) -> Float[Any, "... 4"]:
    _check(convention)
    # Slicing before cos and sin, not after: XLA compiles the per-angle form to faster code.
    half = [euler[..., i : i + 1] / 2 for i in range(3)]
    cos, sin = [xp.cos(h) for h in half], [xp.sin(h) for h in half]
    # Intrinsic sequences compose as first · second · third, extrinsic ones as third · second · first.
    # Either way the product is built from the right, one axis rotation at a time.
    inner, middle, outer = (0, 1, 2) if convention.islower() else (2, 1, 0)
    axes = convention.lower()

    zero = xp.zeros_like(half[0])
    quat = [cos[inner], *(sin[inner] if axes[inner] == name else zero for name in "xyz")]
    for i in (middle, outer):
        quat = _axis_times_quat(cos[i], sin[i], axes[i], *quat)
    # Three float16 products drift by up to 1.5 ulp; renormalizing keeps euler -> rotmat as accurate as one rounding.
    return quaternion.normalize(xp.concatenate(quat, axis=-1), xp)


def _axis_times_quat(c, s, axis: str, w, x, y, z) -> list:
    """Hamilton product ``(c, s·axis) · (w, x, y, z)``: the general product with its zero terms dropped."""
    if axis == "x":
        return [c * w - s * x, c * x + s * w, c * y - s * z, c * z + s * y]
    if axis == "y":
        return [c * w - s * y, c * x + s * z, c * y + s * w, c * z - s * x]
    return [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w]


def quat_to_euler(quat: Float[Any, "... 4"], *, convention: EulerConvention = "ZYX", xp: ModuleType) -> Float[Any, "... 3"]:
    """Bernardes & Viollet (2022), "Quaternion to Euler angles conversion: a direct, general and computationally
    efficient method". At gimbal lock only the sum or difference of the outer angles is defined; like SciPy, the
    third angle is then set to zero.
    """
    _check(convention)
    extrinsic = convention.islower()
    # The method is written for extrinsic sequences; an intrinsic sequence is the reversed extrinsic one.
    sequence = convention.lower() if extrinsic else convention.lower()[::-1]
    i, j, k = ("xyz".index(axis) for axis in sequence)
    proper = i == k
    if proper:
        k = 3 - i - j
    sign = (i - j) * (j - k) * (k - i) // 2  # +1 for a cyclic axis order, -1 otherwise

    # Slices keep unbatched inputs as arrays: NumPy 1.x promotes scalar * Python number to float64.
    w = quat[..., 0:1]
    qi, qj, qk = quat[..., 1 + i : 2 + i], quat[..., 1 + j : 2 + j], sign * quat[..., 1 + k : 2 + k]
    if proper:
        a, b, c, d = w, qi, qj, qk
    else:
        a, b, c, d = w - qj, qi + qk, qj + w, qk - qi

    middle = 2 * xp.arctan2(_hypot(c, d, xp), _hypot(a, b, xp))
    half_sum = _arctan2(b, a, xp)
    half_diff = _arctan2(d, c, xp)

    eps = common.safe_eps(quat.dtype, xp)
    middle_is_zero = middle <= eps
    middle_is_pi = middle >= math.pi - eps
    locked = middle_is_zero | middle_is_pi
    zero = xp.zeros_like(middle)
    first_when_locked = xp.where(middle_is_zero, 2 * half_sum, 2 * half_diff if not extrinsic else -2 * half_diff)
    first = xp.where(locked, first_when_locked, half_sum - half_diff if extrinsic else half_sum + half_diff)
    third = xp.where(locked, zero, half_sum + half_diff if extrinsic else half_sum - half_diff)

    if not proper:
        middle = middle - math.pi / 2
        if extrinsic:
            third = sign * third
        else:
            first = sign * first

    return xp.concatenate([_wrap(first, xp), middle, _wrap(third, xp)], axis=-1)


def _hypot(x: Float[Any, "..."], y: Float[Any, "..."], xp: ModuleType) -> Float[Any, "..."]:
    """sqrt(x^2 + y^2) with a zero (instead of NaN) gradient at the origin."""
    squared = x * x + y * y
    positive = squared > 0
    return xp.where(positive, xp.sqrt(xp.where(positive, squared, xp.ones_like(squared))), xp.zeros_like(squared))


def _arctan2(y: Float[Any, "..."], x: Float[Any, "..."], xp: ModuleType) -> Float[Any, "..."]:
    """arctan2 that returns 0 with a finite gradient at the origin, which gimbal lock reaches exactly."""
    origin = (x == 0) & (y == 0)
    return xp.arctan2(xp.where(origin, xp.zeros_like(y), y), xp.where(origin, xp.ones_like(x), x))


def _wrap(angle: Float[Any, "..."], xp: ModuleType) -> Float[Any, "..."]:
    angle = xp.where(angle > math.pi, angle - 2 * math.pi, angle)
    return xp.where(angle < -math.pi, angle + 2 * math.pi, angle)
