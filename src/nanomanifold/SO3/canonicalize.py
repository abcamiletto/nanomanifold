from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace


def canonicalize(q: Float[Any, "... 4"]) -> Float[Any, "... 4"]:
    xp = get_namespace(q)

    norm = xp.sqrt(xp.sum(q**2, axis=-1, keepdims=True))
    q_normalized = q / norm

    mask = q_normalized[..., 0:1] < 0
    q_canonical = xp.where(mask, -q_normalized, q_normalized)

    return q_canonical
