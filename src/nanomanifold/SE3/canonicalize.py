from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace
from nanomanifold.SO3.canonicalize import canonicalize as canonicalize_quat


def canonicalize(se3: Float[Any, "... 7"]) -> Float[Any, "... 7"]:
    """Canonicalize SE(3) representation by canonicalizing the quaternion part.

    Args:
        se3: SE(3) representation (..., 7) as [w, x, y, z, tx, ty, tz]

    Returns:
        Canonicalized SE(3) representation with quaternion w >= 0
    """
    xp = get_namespace(se3)

    quat = se3[..., :4]
    translation = se3[..., 4:7]

    quat_canonical = canonicalize_quat(quat)

    return xp.concatenate([quat_canonical, translation], axis=-1)
