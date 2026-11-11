"""Group operations on SE(3): identity, composition, inverse and action on points."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import SO3, common
from nanomanifold.common import get_namespace
from nanomanifold.SO3.primitives.quaternion import QuaternionConvention, _from_wxyz, conjugate, from_quat, positive_w, product, rotate


def identity_as(
    ref: Float[Any, "..."],
    *,
    batch_dims: tuple[int, ...],
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 7"]:
    """Identity transforms of shape ``(*batch_dims, 7)``, on ``ref``'s backend, dtype and device."""
    if xp is None:
        xp = get_namespace(ref)
    q = SO3.identity_as(ref, batch_dims=batch_dims, rep_kwargs={"convention": convention}, xp=xp)
    return xp.concatenate([q, common.zeros_as(ref, shape=batch_dims + (3,), xp=xp)], axis=-1)


def multiply(
    T1: Float[Any, "... 7"],
    T2: Float[Any, "... 7"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 7"]:
    """Compose transforms like 4x4 matrices, ``M(T1) @ M(T2)``: ``T2`` is applied first."""
    if xp is None:
        xp = get_namespace(T1)
    q1 = from_quat(T1[..., :4], convention=convention, xp=xp)
    q2 = from_quat(T2[..., :4], convention=convention, xp=xp)
    q = _from_wxyz(positive_w(product(q1, q2, xp), xp), convention=convention, xp=xp)
    t = rotate(q1, T2[..., None, 4:7], xp)[..., 0, :] + T1[..., 4:7]
    return xp.concatenate([q, t], axis=-1)


def inverse(T: Float[Any, "... 7"], *, convention: QuaternionConvention = "wxyz", xp: ModuleType | None = None) -> Float[Any, "... 7"]:
    """Inverse transforms ``(..., 7)``: ``[q, t]`` becomes ``[q⁻¹, -(q⁻¹ t)]``."""
    if xp is None:
        xp = get_namespace(T)
    q_inv = conjugate(from_quat(T[..., :4], convention=convention, xp=xp), xp)
    t_inv = -rotate(q_inv, T[..., None, 4:7], xp)[..., 0, :]
    return xp.concatenate([_from_wxyz(q_inv, convention=convention, xp=xp), t_inv], axis=-1)


def transform_points(
    T: Float[Any, "... 7"],
    points: Float[Any, "... N 3"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... N 3"]:
    """Apply transforms ``(..., 7)`` to points ``(..., N, 3)``: rotate, then translate."""
    if xp is None:
        xp = get_namespace(T)
    return SO3.rotate_points(T[..., :4], points, convention=convention, xp=xp) + T[..., None, 4:7]
