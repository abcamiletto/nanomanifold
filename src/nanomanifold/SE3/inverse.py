from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace
from nanomanifold.SO3 import inverse as so3_inverse
from nanomanifold.SO3 import rotate_points
from nanomanifold.SO3.primitives.quaternion import QuaternionConvention

from .canonicalize import canonicalize


def inverse(
    se3: Float[Any, "... 7"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 7"]:
    """Compute the inverse of SE(3) transformations.

    For an SE(3) transformation T = [R, t] represented as [q, t],
    the inverse is T^(-1) = [R^T, -R^T * t] represented as [q^(-1), -q^(-1) * t].

    Args:
        se3: SE(3) transformation with quaternion components in the given convention
        convention: Quaternion component order, either ``"wxyz"`` or ``"xyzw"``
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        Inverse SE(3) transformation
    """
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace(se3)

    se3 = canonicalize(se3, convention=convention, xp=xp)

    q = se3[..., :4]
    t = se3[..., 4:7]

    q_inv = so3_inverse(q, convention=convention, xp=xp)

    t_inv = -rotate_points(q_inv, t[..., None, :], convention=convention, xp=xp).squeeze(-2)

    result = xp.concatenate([q_inv, t_inv], axis=-1)

    return canonicalize(result, convention=convention, xp=xp)
