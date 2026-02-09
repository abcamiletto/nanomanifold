from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace
from nanomanifold.SO3 import slerp as so3_slerp

from .canonicalize import canonicalize


def slerp(
    se3_1: Float[Any, "... 7"],
    se3_2: Float[Any, "... 7"],
    t: Float[Any, "... N"],
    *,
    xp: ModuleType | None = None,
) -> Float[Any, "... N 7"]:
    """Interpolate between two SE(3) transformations.

    Uses SO3.slerp for the rotation part and linear interpolation for the
    translation part. The interpolation parameter ``t`` follows the same
    semantics as SO3.slerp: output shape is ``(..., N, 7)``.

    Args:
        se3_1: Start SE(3) transformation in [w, x, y, z, tx, ty, tz] format.
        se3_2: End SE(3) transformation in [w, x, y, z, tx, ty, tz] format.
        t: Interpolation parameters. Last dimension N gives the number of
            interpolation points. t=0 returns se3_1, t=1 returns se3_2.
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        Interpolated SE(3) transformations with shape (..., N, 7).
    """
    if xp is None:
        xp = get_namespace(se3_1)

    se3_1 = canonicalize(se3_1, xp=xp)
    se3_2 = canonicalize(se3_2, xp=xp)

    q1 = se3_1[..., :4]
    t1 = se3_1[..., 4:7]
    q2 = se3_2[..., :4]
    t2 = se3_2[..., 4:7]

    # Slerp for rotation: (..., N, 4)
    q_interp = so3_slerp(q1, q2, t, xp=xp)

    # Linear interpolation for translation: (..., N, 3)
    t_expanded = t[..., None]  # (..., N, 1)
    t1_expanded = t1[..., None, :]  # (..., 1, 3)
    t2_expanded = t2[..., None, :]  # (..., 1, 3)
    t_interp = (1.0 - t_expanded) * t1_expanded + t_expanded * t2_expanded

    return xp.concatenate([q_interp, t_interp], axis=-1)
