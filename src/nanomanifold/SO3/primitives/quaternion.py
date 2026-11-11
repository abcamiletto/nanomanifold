"""Quaternion convention helpers for SO(3), and the kernels shared by the quaternion operations.

Public functions canonicalize each quaternion input once on entry (``from_quat``) and reorder their output once on
exit (``_from_wxyz``). In between, everything works on canonical ``wxyz`` quaternions (unit norm, ``w >= 0``), and the
kernels below take and return them without renormalizing.
"""

from types import ModuleType
from typing import Any, Literal

from jaxtyping import Float

from nanomanifold import common
from nanomanifold.common import get_namespace

QuaternionConvention = Literal["wxyz", "xyzw"]


def _check(convention: str) -> None:
    assert convention in ("wxyz", "xyzw"), f"Quaternion convention must be 'wxyz' or 'xyzw', got {convention!r}."


def _to_wxyz(quat: Float[Any, "... 4"], *, convention: str = "wxyz", xp: ModuleType) -> Float[Any, "... 4"]:
    _check(convention)
    if convention == "wxyz":
        return quat
    return xp.concatenate([quat[..., 3:4], quat[..., 0:3]], axis=-1)


def _from_wxyz(quat: Float[Any, "... 4"], *, convention: str = "wxyz", xp: ModuleType) -> Float[Any, "... 4"]:
    _check(convention)
    if convention == "wxyz":
        return quat
    return xp.concatenate([quat[..., 1:4], quat[..., 0:1]], axis=-1)


def normalize(quat: Float[Any, "... 4"], xp: ModuleType) -> Float[Any, "... 4"]:
    """Scale a wxyz quaternion to unit norm and flip it to ``w >= 0``."""
    eps = xp.asarray(common.safe_eps(quat.dtype, xp), dtype=quat.dtype)
    norm = xp.maximum(xp.sqrt(xp.sum(quat * quat, axis=-1, keepdims=True)), eps)
    # The sign flip goes into the (..., 1) divisor, so the (..., 4) array is touched once.
    return quat / xp.where(quat[..., 0:1] < 0, -norm, norm)


def positive_w(quat: Float[Any, "... 4"], xp: ModuleType) -> Float[Any, "... 4"]:
    """Flip a unit wxyz quaternion to the ``w >= 0`` hemisphere."""
    return xp.where(quat[..., 0:1] < 0, -quat, quat)


def conjugate(quat: Float[Any, "... 4"], xp: ModuleType) -> Float[Any, "... 4"]:
    """Inverse of a unit wxyz quaternion; canonical in, canonical out."""
    return xp.concatenate([quat[..., 0:1], -quat[..., 1:4]], axis=-1)


def product(q1: Float[Any, "... 4"], q2: Float[Any, "... 4"], xp: ModuleType) -> Float[Any, "... 4"]:
    """Hamilton product ``q1 q2`` of wxyz quaternions; unit in, unit out, with any sign of ``w``."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return xp.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def rotate(quat: Float[Any, "... 4"], points: Float[Any, "... N 3"], xp: ModuleType) -> Float[Any, "... N 3"]:
    """Rotate points by unit wxyz quaternions: ``p + 2 v × (v × p + w p)`` with ``v`` the vector part."""
    w, x, y, z = quat[..., None, 0], quat[..., None, 1], quat[..., None, 2], quat[..., None, 3]
    px, py, pz = points[..., 0], points[..., 1], points[..., 2]
    # t = v × p + w p stays in components, so the outer cross product does not split it again.
    tx = (y * pz - z * py) + w * px
    ty = (z * px - x * pz) + w * py
    tz = (x * py - y * px) + w * pz
    return points + 2 * xp.stack([y * tz - z * ty, z * tx - x * tz, x * ty - y * tx], axis=-1)


def canonicalize(q: Float[Any, "... 4"], *, convention: QuaternionConvention = "wxyz", xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    """Scale quaternions ``(..., 4)`` to unit norm and flip them to ``w >= 0``, keeping ``convention``."""
    if xp is None:
        xp = get_namespace(q)
    return _from_wxyz(from_quat(q, convention=convention, xp=xp), convention=convention, xp=xp)


def from_quat(q: Float[Any, "... 4"], *, convention: QuaternionConvention = "wxyz", xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    """Canonical ``wxyz`` quaternions ``(..., 4)`` from quaternions in ``convention``."""
    if xp is None:
        xp = get_namespace(q)
    return normalize(_to_wxyz(q, convention=convention, xp=xp), xp)


def to_quat(q: Float[Any, "... 4"], *, convention: QuaternionConvention = "wxyz", xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    """Canonical quaternions ``(..., 4)`` in ``convention`` from ``wxyz`` quaternions."""
    if xp is None:
        xp = get_namespace(q)
    return _from_wxyz(normalize(q, xp), convention=convention, xp=xp)


def largest_eigenvector(matrix: Float[Any, "... 4 4"], *, xp: ModuleType, steps: int = 15, eps: float = 1e-7) -> Float[Any, "... 4"]:
    """Canonical unit eigenvector of the largest eigenvalue of symmetric 4x4 matrices.

    Uses repeated squaring rather than ``eigh``: it is cheap at this size, works in float16, and keeps gradients
    finite when the remaining eigenvalues are degenerate, e.g. when averaging identical rotations.
    """
    one = xp.ones_like(matrix[..., 0, 0])
    zero = xp.zeros_like(one)
    rows = [[one if row == col else zero for col in range(4)] for row in range(4)]
    identity = xp.stack([xp.stack(row, axis=-1) for row in rows], axis=-2)

    # Shift so the largest algebraic eigenvalue also has the largest magnitude, then square the whole matrix
    # instead of iterating a single vector, so no starting guess can be orthogonal to the answer.
    # Halving puts the eigenvalues in (0, 1) and the largest in [1/4, 1), or in [1/2, 1] after each renormalization,
    # so two squarings keep it above 2^-8 and the norm (whose squares must not underflow float16) can skip every other.
    scale = xp.linalg.norm(matrix, axis=(-2, -1), keepdims=True) + eps
    projector = (matrix + scale * identity) / (2 * scale)
    for step in range(steps):
        if step % 2 == 0 and step > 0:
            projector = projector / xp.linalg.norm(projector, axis=(-2, -1), keepdims=True)
        projector = xp.matmul(projector, projector)

    # The projector is close to v v^T; its column with the largest diagonal entry is the best-conditioned copy of v.
    diagonal = xp.stack([projector[..., i, i] for i in range(4)], axis=-1)
    column = xp.argmax(diagonal, axis=-1)
    seed = xp.stack([xp.where(column == i, one, zero) for i in range(4)], axis=-1)
    return normalize(xp.matmul(projector, seed[..., None])[..., 0], xp)


__all__ = ["QuaternionConvention", "canonicalize", "from_quat", "largest_eigenvector", "to_quat"]
