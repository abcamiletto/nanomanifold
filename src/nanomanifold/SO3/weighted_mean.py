from types import ModuleType
from typing import Any, Sequence

from jaxtyping import Float

from nanomanifold import common
from nanomanifold.common import get_namespace

from .primitives.quaternion import QuaternionConvention, from_quat, largest_eigenvector, to_quat


def weighted_mean(
    quaternions: Sequence[Float[Any, "... 4"]],
    weights: Float[Any, "... N"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    """Compute the weighted mean of SO(3) rotations represented as quaternions.

    This function implements the Riemannian mean on SO(3) by computing the weighted
    average in quaternion space using the outer product method. The result is the
    eigenvector corresponding to the largest eigenvalue of the weighted covariance matrix.

    Args:
        quaternions: Sequence of quaternions in the given convention. Each quaternion
            should have shape [..., 4] where the last dimension contains the
            quaternion components.
        weights: Array of weights with shape [..., N] where N is the number of quaternions.
            The weights are normalized internally.
        convention: Quaternion component order, either ``"wxyz"`` or ``"xyzw"``
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        Weighted mean quaternion with shape [..., 4] in the requested convention.
        The result is canonicalized to ensure w >= 0.

    Note:
        This implementation follows the algorithm from:
        "Averaging Quaternions" by F. Landis Markley et al.
    """
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace(quaternions[0])
    original_dtype = quaternions[0].dtype

    quats = xp.stack(
        [from_quat(common.astype(q, original_dtype, xp), convention=convention, xp=xp) for q in quaternions],
        axis=-2,
    )
    weights_array = common.astype(weights, original_dtype, xp)

    weights_normalized = weights_array / xp.sum(weights_array, axis=-1, keepdims=True)
    weighted_quats = weights_normalized[..., :, None] * quats
    M = xp.einsum("...nj,...nk->...jk", weighted_quats, quats)

    return to_quat(largest_eigenvector(M, xp=xp), convention=convention, xp=xp)


def mean(
    quaternions: Sequence[Float[Any, "... 4"]],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    """Compute the mean of SO(3) rotations represented as quaternions.

    This is equivalent to weighted_mean with uniform weights.

    Args:
        quaternions: Sequence of quaternions in the given convention. Each quaternion
            should have shape [..., 4] where the last dimension contains the
            quaternion components.
        convention: Quaternion component order, either ``"wxyz"`` or ``"xyzw"``
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        Mean quaternion with shape [..., 4] in the requested convention.
        The result is canonicalized to ensure w >= 0.
    """
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if len(quaternions) == 0:
        raise ValueError("Cannot compute mean of empty quaternion sequence")

    if xp is None:
        xp = get_namespace(quaternions[0])

    batch_shape = quaternions[0].shape[:-1]
    num_quats = len(quaternions)
    weights = xp.broadcast_to(xp.ones_like(quaternions[0][..., :1]), batch_shape + (num_quats,))

    return weighted_mean(quaternions, weights, convention=convention, xp=xp)
