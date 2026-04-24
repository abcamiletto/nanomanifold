from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold.SO3 import rotate_points
from nanomanifold.SO3.primitives.quaternion import QuaternionConvention

from .canonicalize import canonicalize


def transform_points(
    se3: Float[Any, "... 7"],
    points: Float[Any, "... N 3"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... N 3"]:
    """Transform 3D points using SE(3) transformation.

    Applies both rotation and translation: p' = R * p + t
    where SE(3) = [q, t] with q being the quaternion and t being the translation.

    Args:
        se3: SE(3) transformation with quaternion components in the given convention
        points: Points to transform of shape (..., N, 3)
        convention: Quaternion component order, either ``"wxyz"`` or ``"xyzw"``
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        Transformed points of shape (..., N, 3)
    """
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    se3 = canonicalize(se3, convention=convention, xp=xp)

    q = se3[..., :4]
    t = se3[..., 4:7]

    rotated_points = rotate_points(q, points, convention=convention, xp=xp)

    t_expanded = t[..., None, :]

    transformed_points = rotated_points + t_expanded

    return transformed_points
