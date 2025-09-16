from typing import Any

from jaxtyping import Float

from ..common import get_namespace


def vee(W: Float[Any, "... 3 3"]) -> Float[Any, "... 3"]:
    """Map skew-symmetric matrix to vector (vee operator).

    Args:
        W: (..., 3, 3) skew-symmetric matrices

    Returns:
        (..., 3) tangent vectors in so(3)
    """
    xp = get_namespace(W)

    # Extract components from skew-symmetric matrix:
    # [[ 0, -w3,  w2],
    #  [ w3,  0, -w1],
    #  [-w2,  w1,  0]]

    w1 = W[..., 2, 1]
    w2 = W[..., 0, 2]
    w3 = W[..., 1, 0]

    return xp.stack([w1, w2, w3], axis=-1)
