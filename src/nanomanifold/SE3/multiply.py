from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace
from nanomanifold.SO3 import multiply as so3_multiply
from nanomanifold.SO3 import rotate_points

from .canonicalize import canonicalize


def multiply(se3_1: Float[Any, "... 7"], se3_2: Float[Any, "... 7"]) -> Float[Any, "... 7"]:
    """Multiply two SE(3) transformations.

    The multiplication order matches transformation matrix multiplication:
    multiply(se3_1, se3_2) represents the same composition as to_matrix(se3_1) @ to_matrix(se3_2)

    This means se3_2 is applied first, then se3_1.

    For SE(3) transformations [q1, t1] and [q2, t2], the result is:
    - Quaternion: q1 * q2 (quaternion multiplication)
    - Translation: R1 * t2 + t1 (where R1 rotates by q1)

    Args:
        se3_1: First SE(3) transformation in [w, x, y, z, tx, ty, tz] format
        se3_2: Second SE(3) transformation in [w, x, y, z, tx, ty, tz] format

    Returns:
        Product SE(3) transformation representing the composed transformation
    """
    xp = get_namespace(se3_1)

    se3_1 = canonicalize(se3_1)
    se3_2 = canonicalize(se3_2)

    q1 = se3_1[..., :4]
    t1 = se3_1[..., 4:7]
    q2 = se3_2[..., :4]
    t2 = se3_2[..., 4:7]

    q_result = so3_multiply(q1, q2)

    t2_rotated = rotate_points(q1, t2[..., None, :]).squeeze(-2)
    t_result = t2_rotated + t1

    result = xp.concatenate([q_result, t_result], axis=-1)

    return canonicalize(result)
