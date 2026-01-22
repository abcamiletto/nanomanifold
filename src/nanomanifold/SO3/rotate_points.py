from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace

from .conversions.quaternion import canonicalize


def rotate_points(q: Float[Any, "... 4"], points: Float[Any, "... N 3"]) -> Float[Any, "... N 3"]:
    """Rotate points using quaternion rotation.

    Args:
        q: Quaternion in [w, x, y, z] format of shape (..., 4)
        points: Points to rotate of shape (..., N, 3)

    Returns:
        Rotated points of shape (..., N, 3)
    """
    xp = get_namespace(q)
    q = canonicalize(q)

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    w = w[..., None]
    x = x[..., None]
    y = y[..., None]
    z = z[..., None]

    px, py, pz = points[..., 0], points[..., 1], points[..., 2]

    # Apply quaternion rotation using the formula:
    # p' = q * p * q^(-1)
    # This can be computed efficiently as:
    # p' = p + 2 * cross(q_vec, cross(q_vec, p) + w * p)
    # where q_vec = [x, y, z]

    cross1_x = y * pz - z * py + w * px
    cross1_y = z * px - x * pz + w * py
    cross1_z = x * py - y * px + w * pz

    cross2_x = y * cross1_z - z * cross1_y
    cross2_y = z * cross1_x - x * cross1_z
    cross2_z = x * cross1_y - y * cross1_x

    result_x = px + 2 * cross2_x
    result_y = py + 2 * cross2_y
    result_z = pz + 2 * cross2_z

    return xp.stack([result_x, result_y, result_z], axis=-1)
