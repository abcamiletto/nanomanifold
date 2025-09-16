from nanomanifold.common import get_namespace


def from_rt(quat, translation):
    """Create SE(3) representation from rotation quaternion and translation.

    Args:
        quat: Rotation quaternion (..., 4) as [w, x, y, z]
        translation: Translation vector (..., 3)

    Returns:
        SE(3) representation (..., 7) as [w, x, y, z, tx, ty, tz]
    """
    xp = get_namespace(quat)
    return xp.concatenate([quat, translation], axis=-1)


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
