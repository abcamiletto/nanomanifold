from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace
from nanomanifold.SO3 import inverse as so3_inverse
from nanomanifold.SO3 import rotate_points

from .canonicalize import canonicalize


def inverse(se3: Float[Any, "... 7"]) -> Float[Any, "... 7"]:
    """Compute the inverse of SE(3) transformations.

    For an SE(3) transformation T = [R, t] represented as [q, t],
    the inverse is T^(-1) = [R^T, -R^T * t] represented as [q^(-1), -q^(-1) * t].

    Args:
        se3: SE(3) transformation in [w, x, y, z, tx, ty, tz] format

    Returns:
        Inverse SE(3) transformation
    """
    xp = get_namespace(se3)

    se3 = canonicalize(se3)

    q = se3[..., :4]
    t = se3[..., 4:7]

    q_inv = so3_inverse(q)

    t_inv = -rotate_points(q_inv, t[..., None, :]).squeeze(-2)

    result = xp.concatenate([q_inv, t_inv], axis=-1)

    return canonicalize(result)
