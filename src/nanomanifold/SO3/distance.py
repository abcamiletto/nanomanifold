from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace


def distance(q1: Float[Any, "... 4"], q2: Float[Any, "... 4"]) -> Float[Any, "..."]:
    """Compute the angular distance between two quaternions representing SO(3) rotations.

    The angular distance is the smallest angle needed to rotate from one orientation
    to another, measured in radians. This is equivalent to the geodesic distance
    on the SO(3) manifold.

    Args:
        q1: First quaternion in [w, x, y, z] format
        q2: Second quaternion in [w, x, y, z] format

    Returns:
        Angular distance in radians, in range [0, π]
    """
    xp = get_namespace(q1)

    norm1 = xp.sqrt(xp.sum(q1**2, axis=-1, keepdims=True))
    norm2 = xp.sqrt(xp.sum(q2**2, axis=-1, keepdims=True))
    q1_unit = q1 / norm1
    q2_unit = q2 / norm2

    # Flip sign of q2 when dot(q1, q2) < 0 so the relative rotation
    # always measures the shorter geodesic on the double cover.
    dot_keepdims = xp.sum(q1_unit * q2_unit, axis=-1, keepdims=True)
    q2_unit = xp.where(dot_keepdims < 0, -q2_unit, q2_unit)

    w1 = q1_unit[..., :1]
    v1 = q1_unit[..., 1:]
    w2 = q2_unit[..., :1]
    v2 = q2_unit[..., 1:]

    cross = xp.stack(
        [
            v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
            v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
            v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0],
        ],
        axis=-1,
    )

    vec = w1 * v2 - w2 * v1 - cross
    vec_norm = xp.sqrt(xp.sum(vec**2, axis=-1))
    w = w1 * w2 + xp.sum(v1 * v2, axis=-1, keepdims=True)

    return 2.0 * xp.atan2(vec_norm, w[..., 0])
