"""Uniform random sampling on SO(3) via Shoemake's method."""

import math
from types import ModuleType
from typing import Any

import numpy as np
from jaxtyping import Float

from nanomanifold.common import random_uniform

from .primitives.quaternion import QuaternionConvention, _from_wxyz, positive_w


def random(
    *shape: int,
    dtype=None,
    key=None,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    """Uniformly distributed rotations of shape ``(*shape, 4)``, on NumPy unless ``xp`` is given.

    ``dtype`` defaults to the backend's; JAX needs a PRNG ``key``.
    """
    if xp is None:
        xp = np
    u = random_uniform((*shape, 3), dtype=dtype, key=key, xp=xp)
    return _from_wxyz(shoemake(u, xp=xp), convention=convention, xp=xp)


def shoemake(u: Float[Any, "... 3"], *, xp: ModuleType) -> Float[Any, "... 4"]:
    """Map uniform samples in ``[0, 1)`` to uniformly distributed canonical wxyz quaternions."""
    # Slices keep unbatched samples as arrays: NumPy 1.x promotes scalar * Python float to float64.
    u0, u1, u2 = u[..., 0:1], u[..., 1:2], u[..., 2:3]
    r1 = xp.sqrt(1.0 - u0)
    r2 = xp.sqrt(u0)
    theta1 = 2.0 * math.pi * u1
    theta2 = 2.0 * math.pi * u2

    # Unit by construction: r1² + r2² = 1.
    q = xp.concatenate([r2 * xp.cos(theta2), r1 * xp.sin(theta1), r1 * xp.cos(theta1), r2 * xp.sin(theta2)], axis=-1)
    return positive_w(q, xp)
