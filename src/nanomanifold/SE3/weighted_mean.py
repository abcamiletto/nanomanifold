from types import ModuleType
from typing import Any, Sequence

from jaxtyping import Float

from nanomanifold.common import get_namespace
from nanomanifold.SO3 import weighted_mean as so3_weighted_mean
from nanomanifold.SO3.primitives.quaternion import QuaternionConvention

from .canonicalize import canonicalize


def weighted_mean(
    transforms: Sequence[Float[Any, "... 7"]],
    weights: Float[Any, "... N"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 7"]:
    """Compute the weighted mean of SE(3) transformations.

    Uses SO3.weighted_mean for the rotation part and a weighted average
    for the translation part.

    Args:
        transforms: Sequence of SE(3) transformations with quaternion components in the given convention.
        weights: Array of weights with shape [..., N] where N is the number of transforms.
        convention: Quaternion component order, either ``"wxyz"`` or ``"xyzw"``
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        Weighted mean SE(3) transformation with shape [..., 7].
    """
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace(transforms[0])

    transforms_canon = [canonicalize(t, convention=convention, xp=xp) for t in transforms]
    quaternions = [t[..., :4] for t in transforms_canon]
    translations = [t[..., 4:7] for t in transforms_canon]

    # Weighted mean of rotations
    q_mean = so3_weighted_mean(quaternions, weights, convention=convention, xp=xp)

    # Weighted average of translations
    original_dtype = transforms[0].dtype
    weights_array = xp.asarray(weights, dtype=original_dtype)
    weights_normalized = weights_array / xp.sum(weights_array, axis=-1, keepdims=True)

    t_stack = xp.stack(translations, axis=-2)  # (..., N, 3)
    t_mean = xp.sum(weights_normalized[..., :, None] * t_stack, axis=-2)  # (..., 3)

    return xp.concatenate([q_mean, t_mean], axis=-1)


def mean(
    transforms: Sequence[Float[Any, "... 7"]],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 7"]:
    """Compute the mean of SE(3) transformations.

    Equivalent to weighted_mean with uniform weights.

    Args:
        transforms: Sequence of SE(3) transformations with quaternion components in the given convention.
        convention: Quaternion component order, either ``"wxyz"`` or ``"xyzw"``
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        Mean SE(3) transformation with shape [..., 7].
    """
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if len(transforms) == 0:
        raise ValueError("Cannot compute mean of empty transform sequence")

    if xp is None:
        xp = get_namespace(transforms[0])

    batch_shape = transforms[0].shape[:-1]
    num_transforms = len(transforms)
    weights = xp.broadcast_to(xp.ones_like(transforms[0][..., :1]), batch_shape + (num_transforms,))

    return weighted_mean(transforms, weights, convention=convention, xp=xp)
