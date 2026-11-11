"""Uniform random sampling on SE(3)."""

from types import ModuleType
from typing import Any

import numpy as np
from jaxtyping import Float

from nanomanifold.common import random_uniform
from nanomanifold.SO3.primitives.quaternion import QuaternionConvention, _from_wxyz
from nanomanifold.SO3.random import shoemake


def random(
    *shape: int,
    dtype=None,
    key=None,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 7"]:
    """Random transforms ``(*shape, 7)``: uniform rotations and translations uniform in ``[-1, 1)³``.

    Uses NumPy unless ``xp`` is given; ``dtype`` defaults to the backend's; JAX needs a PRNG ``key``.
    """
    if xp is None:
        xp = np

    # Single draw: first 3 for Shoemake rotation, last 3 for translation
    u = random_uniform((*shape, 6), dtype=dtype, key=key, xp=xp)
    q = _from_wxyz(shoemake(u[..., :3], xp=xp), convention=convention, xp=xp)
    t = 2.0 * u[..., 3:6] - 1.0  # [0, 1) -> [-1, 1)
    return xp.concatenate([q, t], axis=-1)
