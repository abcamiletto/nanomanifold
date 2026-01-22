from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace

from .canonicalize import canonicalize
from .conversions.quaternion import from_quat_xyzw, to_quat_xyzw


def multiply(q1: Float[Any, "... 4"], q2: Float[Any, "... 4"], xyzw: bool = False) -> Float[Any, "... 4"]:
    """Multiply two quaternions representing SO(3) rotations.

    The multiplication order matches rotation matrix multiplication:
    multiply(q1, q2) represents the same composition as to_matrix(q1) @ to_matrix(q2)

    This means q2 is applied first, then q1.

    Args:
        q1: First quaternion in [w, x, y, z] format (or [x, y, z, w] if xyzw=True)
        q2: Second quaternion in [w, x, y, z] format (or [x, y, z, w] if xyzw=True)
        xyzw: Whether to interpret inputs (and return output) as [x, y, z, w]

    Returns:
        Product quaternion representing the composed rotation
    """
    xp = get_namespace(q1)

    if xyzw:
        q1 = from_quat_xyzw(q1)
        q2 = from_quat_xyzw(q2)

    q1 = canonicalize(q1)
    q2 = canonicalize(q2)

    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    result = xp.stack([w, x, y, z], axis=-1)
    result = canonicalize(result)

    if xyzw:
        return to_quat_xyzw(result)
    return result
