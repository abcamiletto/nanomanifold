from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace
from nanomanifold.SO3.conversions.quaternion import from_quat_xyzw, to_quat_xyzw


def canonicalize(q: Float[Any, "... 4"], xyzw: bool = False) -> Float[Any, "... 4"]:
    xp = get_namespace(q)
    if xyzw:
        q = from_quat_xyzw(q)

    norm = xp.sqrt(xp.sum(q**2, axis=-1, keepdims=True))
    q_normalized = q / norm

    mask = q_normalized[..., 0:1] < 0
    q_canonical = xp.where(mask, -q_normalized, q_normalized)

    if xyzw:
        return to_quat_xyzw(q_canonical)
    return q_canonical
