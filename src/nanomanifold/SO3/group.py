"""Group operations on SO(3): identity, composition, inverse, action on points and distance."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import common
from nanomanifold.common import get_namespace

from .convert import RotationRep, convert
from .primitives import euler, matrix
from .primitives.quaternion import QuaternionConvention, _from_wxyz, conjugate, from_quat, positive_w, product, rotate


def identity_as(
    ref: Float[Any, "..."],
    *,
    batch_dims: tuple[int, ...],
    rep: RotationRep = "quat",
    rep_kwargs: dict[str, Any] = {},
    xp: ModuleType | None = None,
) -> Float[Any, "..."]:
    """Identity rotations of shape ``batch_dims`` in representation ``rep``, on ``ref``'s backend, dtype and device.

    ``rep_kwargs`` are the options of ``rep`` as in ``convert``. The ``hinge`` identity is zero whatever the axes.
    """
    if xp is None:
        xp = get_namespace(ref)
    # Assembled from exact zeros and ones instead of converted from the identity quaternion, which would cost
    # more than the SE(3) exp and log that use it. Options are only checked, apart from the quaternion order.
    if rep == "euler":
        euler._check(rep_kwargs.get("convention", "ZYX"))
    if rep == "matrix":
        matrix._check(rep_kwargs.get("mode", "svd"))
    if rep in ("matrix", "rotmat"):
        device = {"device": ref.device} if "torch" in xp.__name__ else {}
        return xp.broadcast_to(xp.eye(3, dtype=ref.dtype, **device), batch_dims + (3, 3))
    if rep not in _IDENTITY_VECTORS:
        raise ValueError(f"Unsupported rep {rep!r}.")
    zero = common.zeros_as(ref, shape=batch_dims + (1,), xp=xp)
    one = zero + 1
    identity = xp.concatenate([one if value else zero for value in _IDENTITY_VECTORS[rep]], axis=-1)
    return _from_wxyz(identity, convention=rep_kwargs.get("convention", "wxyz"), xp=xp) if rep == "quat" else identity


_IDENTITY_VECTORS = {"quat": (1, 0, 0, 0), "axis_angle": (0, 0, 0), "euler": (0, 0, 0), "hinge": (0,), "sixd": (1, 0, 0, 0, 1, 0)}


def multiply(
    q1: Float[Any, "... 4"],
    q2: Float[Any, "... 4"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    """Compose rotations like rotation matrices, ``R(q1) @ R(q2)``: ``q2`` is applied first."""
    if xp is None:
        xp = get_namespace(q1)
    q = product(from_quat(q1, convention=convention, xp=xp), from_quat(q2, convention=convention, xp=xp), xp)
    return _from_wxyz(positive_w(q, xp), convention=convention, xp=xp)


def inverse(q: Float[Any, "... 4"], *, convention: QuaternionConvention = "wxyz", xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    """Inverse rotations ``(..., 4)``."""
    if xp is None:
        xp = get_namespace(q)
    return _from_wxyz(conjugate(from_quat(q, convention=convention, xp=xp), xp), convention=convention, xp=xp)


def rotate_points(
    q: Float[Any, "... 4"],
    points: Float[Any, "... N 3"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... N 3"]:
    """Rotate points ``(..., N, 3)`` by quaternions ``(..., 4)``."""
    if xp is None:
        xp = get_namespace(q)
    return rotate(from_quat(q, convention=convention, xp=xp), points, xp)


def distance(
    r1: Float[Any, "..."],
    r2: Float[Any, "..."],
    *,
    rep: RotationRep = "quat",
    rep_kwargs: dict[str, Any] = {},
    xp: ModuleType | None = None,
) -> Float[Any, "..."]:
    """Geodesic distance ``(...)`` in radians, in ``[0, π]``, between rotations given in representation ``rep``.

    ``rep_kwargs`` are the options of ``rep`` as in ``convert``.
    """
    if xp is None:
        xp = get_namespace(r1)
    q1 = convert(r1, src=rep, dst="quat", src_kwargs=rep_kwargs, xp=xp)
    q2 = convert(r2, src=rep, dst="quat", src_kwargs=rep_kwargs, xp=xp)

    # The angle of the relative rotation q1⁻¹ q2 = (w, v) is 2·atan2(|v|, |w|).
    relative = product(conjugate(q1, xp), q2, xp)
    eps = xp.asarray(common.safe_eps(relative.dtype, xp), dtype=relative.dtype)
    eps_sq = eps * eps
    vec_sq = xp.sum(relative[..., 1:] ** 2, axis=-1)
    vec_norm = xp.sqrt(xp.maximum(vec_sq, eps_sq))
    two = xp.ones_like(vec_norm) + xp.ones_like(vec_norm)
    angle = two * xp.arctan2(vec_norm, xp.abs(relative[..., 0]))

    return xp.where(vec_sq <= eps_sq, xp.zeros_like(angle), angle)
