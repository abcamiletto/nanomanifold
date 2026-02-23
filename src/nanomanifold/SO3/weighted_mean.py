from types import ModuleType
from typing import Any, Sequence

from jaxtyping import Float

from nanomanifold import common
from nanomanifold.common import get_namespace

from .primitives.quaternion import canonicalize


def _as_dtype_preserve_grad(x, dtype, xp):
    """Cast while preserving autograd for torch tensors."""
    if "torch" in xp.__name__ and hasattr(x, "to"):
        return x.to(dtype=dtype)
    return xp.asarray(x, dtype=dtype)


def weighted_mean(
    quaternions: Sequence[Float[Any, "... 4"]], weights: Float[Any, "... N"], *, xp: ModuleType | None = None
) -> Float[Any, "... 4"]:
    """Compute the weighted mean of SO(3) rotations represented as quaternions.

    This function implements the Riemannian mean on SO(3) by computing the weighted
    average in quaternion space using the outer product method. The result is the
    eigenvector corresponding to the largest eigenvalue of the weighted covariance matrix.

    Args:
        quaternions: Sequence of quaternions in [w, x, y, z] format. Each quaternion
                    should have shape [..., 4] where the last dimension contains the
                    quaternion components.
        weights: Array of weights with shape [..., N] where N is the number of quaternions.
                The weights are normalized internally.
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        Weighted mean quaternion with shape [..., 4] in [w, x, y, z] format.
        The result is canonicalized to ensure w >= 0.

    Note:
        This implementation follows the algorithm from:
        "Averaging Quaternions" by F. Landis Markley et al.
    """
    if xp is None:
        xp = get_namespace(quaternions[0])
    original_dtype = quaternions[0].dtype

    quats = xp.stack([_as_dtype_preserve_grad(q, original_dtype, xp) for q in quaternions], axis=-2)
    weights_array = _as_dtype_preserve_grad(weights, original_dtype, xp)

    norms = xp.linalg.norm(quats, axis=-1, keepdims=True)
    eps = common.safe_eps(original_dtype, xp)
    safe_norms = xp.where(norms < eps, eps, norms)
    quats_normalized = quats / safe_norms

    sign_mask = quats_normalized[..., 0:1] < 0
    quats_canonical = xp.where(sign_mask, -quats_normalized, quats_normalized)

    weights_normalized = weights_array / xp.sum(weights_array, axis=-1, keepdims=True)

    weighted_quats = weights_normalized[..., :, None] * quats_canonical

    M = xp.einsum("...nj,...nk->...jk", weighted_quats, quats_canonical)

    if original_dtype == xp.float16:
        M_compute = _as_dtype_preserve_grad(M, xp.float32, xp)
        eigenvalues, eigenvectors = xp.linalg.eigh(M_compute)
        eigenvectors = _as_dtype_preserve_grad(eigenvectors, original_dtype, xp)
    else:
        eigenvalues, eigenvectors = xp.linalg.eigh(M)

    avg_quat = eigenvectors[..., :, -1]

    avg_quat = avg_quat / xp.linalg.norm(avg_quat, axis=-1, keepdims=True)

    return canonicalize(avg_quat, xp=xp)


def mean(quaternions: Sequence[Float[Any, "... 4"]], *, xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    """Compute the mean of SO(3) rotations represented as quaternions.

    This is equivalent to weighted_mean with uniform weights.

    Args:
        quaternions: Sequence of quaternions in [w, x, y, z] format. Each quaternion
                    should have shape [..., 4] where the last dimension contains the
                    quaternion components.
        xp: Array namespace (e.g. torch, jax.numpy). If None, auto-detected.

    Returns:
        Mean quaternion with shape [..., 4] in [w, x, y, z] format.
        The result is canonicalized to ensure w >= 0.
    """
    if len(quaternions) == 0:
        raise ValueError("Cannot compute mean of empty quaternion sequence")

    if xp is None:
        xp = get_namespace(quaternions[0])

    batch_shape = quaternions[0].shape[:-1]
    num_quats = len(quaternions)
    weights = xp.broadcast_to(xp.ones_like(quaternions[0][..., :1]), batch_shape + (num_quats,))

    return weighted_mean(quaternions, weights, xp=xp)
