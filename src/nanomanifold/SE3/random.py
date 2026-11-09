"""Uniform random sampling on SE(3)."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace_by_name, random_uniform
from nanomanifold.SO3.primitives.quaternion import QuaternionConvention, to_quat
from nanomanifold.SO3.random import shoemake


def random(
    *shape: int,
    dtype=None,
    key=None,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 7"]:
    """Sample uniformly distributed random rigid transforms on SE(3).

    Rotation is uniform on SO(3) via Shoemake's method.
    Translation is uniform on [-1, 1]^3.

    Args:
        *shape: Batch dimensions (e.g. ``random(10)`` gives 10 transforms).
        dtype: Output dtype. If None, uses backend default.
        key: JAX PRNG key (required when xp is JAX).
        convention: Quaternion component order, either ``"wxyz"`` or ``"xyzw"``
        xp: Array namespace. If None, uses numpy.

    Returns:
        SE(3) arrays with quaternion components in the requested convention, shape ``(*shape, 7)``.
    """
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace_by_name("numpy")

    # Single draw: first 3 for Shoemake rotation, last 3 for translation
    u = random_uniform((*shape, 6), dtype=dtype, key=key, xp=xp)

    q = to_quat(shoemake(u[..., :3], xp=xp), convention=convention, xp=xp)

    # Translation: map [0,1) -> [-1,1)
    t = 2.0 * u[..., 3:6] - 1.0

    return xp.concatenate([q, t], axis=-1)
