"""Rotation-matrix conversions for SO(3) rotations."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import common
from nanomanifold.common import get_namespace

from .quaternion import QuaternionConvention, canonicalize, from_quat, to_quat


def to_rotmat(
    q: Float[Any, "... 4"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 3 3"]:
    """Convert quaternion to a normalized 3x3 rotation matrix."""
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace(q)
    q = from_quat(q, convention=convention, xp=xp)
    q = canonicalize(q, xp=xp)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    one = xp.ones_like(w)
    two = one + one

    return xp.stack(
        [
            xp.stack([one - two * (y * y + z * z), two * (x * y - w * z), two * (x * z + w * y)], axis=-1),
            xp.stack([two * (x * y + w * z), one - two * (x * x + z * z), two * (y * z - w * x)], axis=-1),
            xp.stack([two * (x * z - w * y), two * (y * z + w * x), one - two * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def _project_matrix_to_rotmat_svd(matrix: Float[Any, "... 3 3"], xp) -> Float[Any, "... 3 3"]:
    if matrix.dtype == xp.float16:
        raise ValueError("SO3 matrix projection with mode='svd' does not support float16.")

    u, _, vh = xp.linalg.svd(matrix)
    det = xp.linalg.det(xp.matmul(u, vh))

    zero = xp.zeros_like(det)
    one = xp.ones_like(det)
    sign = xp.where(det < 0, -one, one)
    correction = xp.stack(
        [
            xp.stack([one, zero, zero], axis=-1),
            xp.stack([zero, one, zero], axis=-1),
            xp.stack([zero, zero, sign], axis=-1),
        ],
        axis=-2,
    )
    return xp.matmul(u, xp.matmul(correction, vh))


def _project_matrix_to_rotmat_davenport(matrix: Float[Any, "... 3 3"], xp, *, steps: int = 20, eps: float = 1e-7) -> Float[Any, "... 3 3"]:
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    trace = m00 + m11 + m22
    z = xp.stack([m21 - m12, m02 - m20, m10 - m01], axis=-1)
    k = xp.stack(
        [
            xp.stack([trace, z[..., 0], z[..., 1], z[..., 2]], axis=-1),
            xp.stack([z[..., 0], m00 - m11 - m22, m01 + m10, m02 + m20], axis=-1),
            xp.stack([z[..., 1], m01 + m10, -m00 + m11 - m22, m12 + m21], axis=-1),
            xp.stack([z[..., 2], m02 + m20, m12 + m21, -m00 - m11 + m22], axis=-1),
        ],
        axis=-2,
    )

    one = xp.ones_like(trace)
    zero = xp.zeros_like(trace)
    quat = xp.stack([one, zero, zero, zero], axis=-1)

    for _ in range(steps):
        quat = xp.matmul(k, quat[..., None])[..., 0]
        quat = quat / (xp.linalg.norm(quat, axis=-1, keepdims=True) + eps)

    quat = xp.where(quat[..., :1] < 0, -quat, quat)
    return to_rotmat(quat, xp=xp)


def _project_matrix_to_rotmat(matrix: Float[Any, "... 3 3"], xp, *, mode: str = "svd") -> Float[Any, "... 3 3"]:
    if mode == "svd":
        return _project_matrix_to_rotmat_svd(matrix, xp)
    if mode == "davenport":
        return _project_matrix_to_rotmat_davenport(matrix, xp)
    raise ValueError(f"Unsupported projection mode '{mode}'. Supported values: svd, davenport.")


def from_rotmat(
    rotmat: Float[Any, "... 3 3"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    """Convert a normalized 3x3 rotation matrix to a quaternion."""
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace(rotmat)

    trace = rotmat[..., 0, 0] + rotmat[..., 1, 1] + rotmat[..., 2, 2]

    zero = xp.zeros_like(trace)
    one = xp.ones_like(trace)
    eps = xp.asarray(common.safe_eps(rotmat.dtype, xp), dtype=rotmat.dtype)
    two = one + one
    quarter = one / (two + two)

    s1 = xp.sqrt(xp.maximum(zero + eps, trace + one)) * two
    s1_safe = xp.where(s1 < eps, eps, s1)
    w1 = quarter * s1
    x1 = (rotmat[..., 2, 1] - rotmat[..., 1, 2]) / s1_safe
    y1 = (rotmat[..., 0, 2] - rotmat[..., 2, 0]) / s1_safe
    z1 = (rotmat[..., 1, 0] - rotmat[..., 0, 1]) / s1_safe

    s2 = xp.sqrt(xp.maximum(zero + eps, one + rotmat[..., 0, 0] - rotmat[..., 1, 1] - rotmat[..., 2, 2])) * two
    s2_safe = xp.where(s2 < eps, eps, s2)
    w2 = (rotmat[..., 2, 1] - rotmat[..., 1, 2]) / s2_safe
    x2 = quarter * s2
    y2 = (rotmat[..., 0, 1] + rotmat[..., 1, 0]) / s2_safe
    z2 = (rotmat[..., 0, 2] + rotmat[..., 2, 0]) / s2_safe

    s3 = xp.sqrt(xp.maximum(zero + eps, one + rotmat[..., 1, 1] - rotmat[..., 0, 0] - rotmat[..., 2, 2])) * two
    s3_safe = xp.where(s3 < eps, eps, s3)
    w3 = (rotmat[..., 0, 2] - rotmat[..., 2, 0]) / s3_safe
    x3 = (rotmat[..., 0, 1] + rotmat[..., 1, 0]) / s3_safe
    y3 = quarter * s3
    z3 = (rotmat[..., 1, 2] + rotmat[..., 2, 1]) / s3_safe

    s4 = xp.sqrt(xp.maximum(zero + eps, one + rotmat[..., 2, 2] - rotmat[..., 0, 0] - rotmat[..., 1, 1])) * two
    s4_safe = xp.where(s4 < eps, eps, s4)
    w4 = (rotmat[..., 1, 0] - rotmat[..., 0, 1]) / s4_safe
    x4 = (rotmat[..., 0, 2] + rotmat[..., 2, 0]) / s4_safe
    y4 = (rotmat[..., 1, 2] + rotmat[..., 2, 1]) / s4_safe
    z4 = quarter * s4

    cond1 = trace > 0
    cond2 = (rotmat[..., 0, 0] > rotmat[..., 1, 1]) & (rotmat[..., 0, 0] > rotmat[..., 2, 2])
    cond3 = rotmat[..., 1, 1] > rotmat[..., 2, 2]

    w = xp.where(cond1, w1, xp.where(cond2, w2, xp.where(cond3, w3, w4)))
    x = xp.where(cond1, x1, xp.where(cond2, x2, xp.where(cond3, x3, x4)))
    y = xp.where(cond1, y1, xp.where(cond2, y2, xp.where(cond3, y3, y4)))
    z = xp.where(cond1, z1, xp.where(cond2, z2, xp.where(cond3, z3, z4)))

    quat = canonicalize(xp.stack([w, x, y, z], axis=-1), xp=xp)
    return to_quat(quat, convention=convention, xp=xp)


def from_matrix(
    matrix: Float[Any, "... 3 3"],
    *,
    convention: QuaternionConvention = "wxyz",
    mode: str = "svd",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    """Convert a generic 3x3 matrix to a quaternion by projecting to SO(3)."""
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace(matrix)
    matrix = _project_matrix_to_rotmat(matrix, xp, mode=mode)
    return from_rotmat(matrix, convention=convention, xp=xp)
