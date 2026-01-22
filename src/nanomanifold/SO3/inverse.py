from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace

from .conversions.quaternion import canonicalize


def inverse(q: Float[Any, "... 4"]) -> Float[Any, "... 4"]:
    xp = get_namespace(q)
    q = canonicalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    q_inv = xp.stack([w, -x, -y, -z], axis=-1)

    return canonicalize(q_inv)
