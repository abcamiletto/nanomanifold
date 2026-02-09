"""Uniform random sampling on SE(3)."""

import math
from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace_by_name, random_uniform
from nanomanifold.SO3.conversions.quaternion import canonicalize


def random(*shape: int, dtype=None, key=None, xp: ModuleType | None = None) -> Float[Any, "... 7"]:
    """Sample uniformly distributed random rigid transforms on SE(3).

    Rotation is uniform on SO(3) via Shoemake's method.
    Translation is uniform on [-1, 1]^3.

    Args:
        *shape: Batch dimensions (e.g. ``random(10)`` gives 10 transforms).
        dtype: Output dtype. If None, uses backend default.
        key: JAX PRNG key (required when xp is JAX).
        xp: Array namespace. If None, uses numpy.

    Returns:
        SE(3) arrays in [w, x, y, z, tx, ty, tz] format, shape ``(*shape, 7)``.
    """
    if xp is None:
        xp = get_namespace_by_name("numpy")

    # Single draw: first 3 for Shoemake rotation, last 3 for translation
    u = random_uniform((*shape, 6), dtype=dtype, key=key, xp=xp)

    # Shoemake's method for uniform SO(3)
    u0 = u[..., 0]
    u1 = u[..., 1]
    u2 = u[..., 2]

    r1 = xp.sqrt(1.0 - u0)
    r2 = xp.sqrt(u0)
    theta1 = 2.0 * math.pi * u1
    theta2 = 2.0 * math.pi * u2

    w = r2 * xp.cos(theta2)
    x = r1 * xp.sin(theta1)
    y = r1 * xp.cos(theta1)
    z = r2 * xp.sin(theta2)

    q = xp.stack([w, x, y, z], axis=-1)
    q = canonicalize(q, xp=xp)

    # Translation: map [0,1) -> [-1,1)
    t = 2.0 * u[..., 3:6] - 1.0

    return xp.concatenate([q, t], axis=-1)
