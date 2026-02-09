from types import ModuleType

from nanomanifold.common import get_namespace

from ..canonicalize import canonicalize as canonicalize_se3


def from_rt(quat, translation, *, xp: ModuleType | None = None):
    """Create SE(3) representation from rotation quaternion and translation.

    Args:
        quat: Rotation quaternion (..., 4) as [w, x, y, z]
        translation: Translation vector (..., 3)
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        SE(3) representation (..., 7) as [w, x, y, z, tx, ty, tz] with the
        quaternion canonicalized to have a non-negative scalar component.
    """
    if xp is None:
        xp = get_namespace(quat)
    se3 = xp.concatenate([quat, translation], axis=-1)
    return canonicalize_se3(se3, xp=xp)


def to_rt(se3):
    """Extract rotation quaternion and translation from SE(3) representation.

    Args:
        se3: SE(3) representation (..., 7) as [w, x, y, z, tx, ty, tz]

    Returns:
        quat: Rotation quaternion (..., 4) as [w, x, y, z]
        translation: Translation vector (..., 3)
    """
    quat = se3[..., :4]
    translation = se3[..., 4:7]
    return quat, translation
