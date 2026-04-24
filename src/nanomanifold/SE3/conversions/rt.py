from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace
from nanomanifold.SO3.primitives.quaternion import QuaternionConvention
from nanomanifold.SO3.primitives.quaternion import canonicalize as canonicalize_quat

from ..canonicalize import canonicalize as canonicalize_se3


def from_rt(
    quat: Float[Any, "... 4"],
    translation: Float[Any, "... 3"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 7"]:
    """Create SE(3) representation from rotation quaternion and translation.

    Args:
        quat: Rotation quaternion (..., 4) in the given convention
        translation: Translation vector (..., 3)
        convention: Quaternion component order, either ``"wxyz"`` or ``"xyzw"``
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        SE(3) representation (..., 7) with quaternion components in the given
        convention and canonicalized to have a non-negative scalar component.
    """
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace(quat)
    se3 = xp.concatenate([quat, translation], axis=-1)
    return canonicalize_se3(se3, convention=convention, xp=xp)


def to_rt(
    se3: Float[Any, "... 7"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> tuple[Float[Any, "... 4"], Float[Any, "... 3"]]:
    """Extract rotation quaternion and translation from SE(3) representation.

    Args:
        se3: SE(3) representation (..., 7) with quaternion components in the given convention
        convention: Quaternion component order, either ``"wxyz"`` or ``"xyzw"``

    Returns:
        quat: Rotation quaternion (..., 4) in the requested convention
        translation: Translation vector (..., 3)
    """
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace(se3)
    quat = se3[..., :4]
    translation = se3[..., 4:7]
    return canonicalize_quat(quat, convention=convention, xp=xp), translation
