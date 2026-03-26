from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace
from nanomanifold.SO3.primitives.quaternion import QuaternionConvention
from nanomanifold.SO3.primitives.quaternion import canonicalize as canonicalize_quat


def canonicalize(
    se3: Float[Any, "... 7"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 7"]:
    """Canonicalize SE(3) representation by canonicalizing the quaternion part.

    Args:
        se3: SE(3) representation (..., 7) with quaternion components in the given convention

    Returns:
        Canonicalized SE(3) representation with quaternion w >= 0
    """
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace(se3)

    quat = se3[..., :4]
    translation = se3[..., 4:7]

    quat_canonical = canonicalize_quat(quat, convention=convention, xp=xp)

    return xp.concatenate([quat_canonical, translation], axis=-1)
