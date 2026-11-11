from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace

from .primitives.quaternion import QuaternionConvention, canonicalize, from_quat, to_quat


def multiply(
    q1: Float[Any, "... 4"],
    q2: Float[Any, "... 4"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    """Multiply two quaternions representing SO(3) rotations.

    The multiplication order matches rotation-matrix multiplication:
    multiply(q1, q2) represents the same composition as to_rotmat(q1) @ to_rotmat(q2)

    This means q2 is applied first, then q1.

    Args:
        q1: First quaternion in the given convention
        q2: Second quaternion in the given convention
        convention: Quaternion component order, either ``"wxyz"`` or ``"xyzw"``
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        Product quaternion representing the composed rotation
    """
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace(q1)

    q1 = from_quat(q1, convention=convention, xp=xp)
    q2 = from_quat(q2, convention=convention, xp=xp)
    q1 = canonicalize(q1, xp=xp)
    q2 = canonicalize(q2, xp=xp)

    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    result = xp.stack([w, x, y, z], axis=-1)
    result = canonicalize(result, xp=xp)

    return to_quat(result, convention=convention, xp=xp)
