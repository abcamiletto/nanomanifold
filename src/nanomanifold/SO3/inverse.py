from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace

from .conversions.quaternion import canonicalize


def inverse(q: Float[Any, "... 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    if xp is None:
        xp = get_namespace(q)
    q = canonicalize(q, xp=xp)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    q_inv = xp.stack([w, -x, -y, -z], axis=-1)

    return canonicalize(q_inv, xp=xp)
