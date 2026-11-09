"""Uniform random sampling on SO(3) via Shoemake's method."""

import math
from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace_by_name, random_uniform

from .primitives.quaternion import QuaternionConvention, canonicalize, to_quat


def random(
    *shape: int,
    dtype=None,
    key=None,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    """Sample uniformly distributed random rotations on SO(3).

    Uses Shoemake's method to generate uniform unit quaternions.

    Args:
        *shape: Batch dimensions (e.g. ``random(10)`` gives 10 rotations).
        dtype: Output dtype. If None, uses backend default.
        key: JAX PRNG key (required when xp is JAX).
        convention: Quaternion component order, either ``"wxyz"`` or ``"xyzw"``
        xp: Array namespace. If None, uses numpy.

    Returns:
        Quaternions in the requested convention, shape ``(*shape, 4)``.
    """
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace_by_name("numpy")

    u = random_uniform((*shape, 3), dtype=dtype, key=key, xp=xp)
    return to_quat(shoemake(u, xp=xp), convention=convention, xp=xp)


def shoemake(u: Float[Any, "... 3"], *, xp: ModuleType) -> Float[Any, "... 4"]:
    """Map uniform samples in ``[0, 1)`` to uniformly distributed canonical wxyz quaternions."""
    # Slices keep unbatched samples as arrays: NumPy 1.x promotes scalar * Python float to float64.
    u0, u1, u2 = u[..., 0:1], u[..., 1:2], u[..., 2:3]
    r1 = xp.sqrt(1.0 - u0)
    r2 = xp.sqrt(u0)
    theta1 = 2.0 * math.pi * u1
    theta2 = 2.0 * math.pi * u2

    q = xp.concatenate([r2 * xp.cos(theta2), r1 * xp.sin(theta1), r1 * xp.cos(theta1), r2 * xp.sin(theta2)], axis=-1)
    return canonicalize(q, xp=xp)
