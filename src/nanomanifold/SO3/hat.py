from typing import Any

from jaxtyping import Float

from ..common import get_namespace


def hat(w: Float[Any, "... 3"]) -> Float[Any, "... 3 3"]:
    """Map vector to skew-symmetric matrix (hat operator).

    Args:
        w: (..., 3) array representing tangent vectors in so(3)

    Returns:
        (..., 3, 3) skew-symmetric matrices
    """
    xp = get_namespace(w)

    w1 = w[..., 0]
    w2 = w[..., 1]
    w3 = w[..., 2]

    zero = w1 * 0

    # Build skew-symmetric matrix:
    # [[ 0, -w3,  w2],
    #  [ w3,  0, -w1],
    #  [-w2,  w1,  0]]

    row1 = xp.stack([zero, -w3, w2], axis=-1)
    row2 = xp.stack([w3, zero, -w1], axis=-1)
    row3 = xp.stack([-w2, w1, zero], axis=-1)

    result = xp.stack([row1, row2, row3], axis=-2)

    return result
