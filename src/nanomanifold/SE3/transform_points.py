from typing import Any

from jaxtyping import Float

from nanomanifold.SO3 import rotate_points

from .canonicalize import canonicalize


def transform_points(se3: Float[Any, "... 7"], points: Float[Any, "... N 3"]) -> Float[Any, "... N 3"]:
    """Transform 3D points using SE(3) transformation.

    Applies both rotation and translation: p' = R * p + t
    where SE(3) = [q, t] with q being the quaternion and t being the translation.

    Args:
        se3: SE(3) transformation in [w, x, y, z, tx, ty, tz] format of shape (..., 7)
        points: Points to transform of shape (..., N, 3)

    Returns:
        Transformed points of shape (..., N, 3)
    """
    se3 = canonicalize(se3)

    q = se3[..., :4]
    t = se3[..., 4:7]

    rotated_points = rotate_points(q, points)

    t_expanded = t[..., None, :]

    transformed_points = rotated_points + t_expanded

    return transformed_points
