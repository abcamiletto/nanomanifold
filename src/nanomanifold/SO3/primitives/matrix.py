"""Matrix edges: matrix <-> rotmat. A matrix is any 3x3 array; it becomes a rotation by projection onto SO(3)."""

from types import ModuleType
from typing import Any, Literal

from jaxtyping import Float

from . import quaternion, rotmat

ProjectionMode = Literal["svd", "davenport"]


def _check(mode: str) -> None:
    if mode not in ("svd", "davenport"):
        raise ValueError(f"Unsupported projection mode '{mode}'. Supported values: svd, davenport.")


def matrix_to_rotmat(matrix: Float[Any, "... 3 3"], *, mode: ProjectionMode = "svd", xp: ModuleType) -> Float[Any, "... 3 3"]:
    _check(mode)
    return _svd(matrix, xp) if mode == "svd" else _davenport(matrix, xp)


def rotmat_to_matrix(rotmat: Float[Any, "... 3 3"], *, mode: ProjectionMode = "svd", xp: ModuleType) -> Float[Any, "... 3 3"]:
    """A rotation matrix already is one; ``mode`` only matters when reading a matrix, so it is just checked."""
    _check(mode)
    return rotmat


def _svd(matrix: Float[Any, "... 3 3"], xp: ModuleType) -> Float[Any, "... 3 3"]:
    if matrix.dtype == xp.float16:
        raise ValueError("SO3 matrix projection with mode='svd' does not support float16.")

    u, _, vh = xp.linalg.svd(matrix, full_matrices=False)  # reduced form: old JAX cannot differentiate the full one
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


def _davenport(matrix: Float[Any, "... 3 3"], xp: ModuleType) -> Float[Any, "... 3 3"]:
    """The rotation maximizing tr(R^T M) is the top eigenvector of Davenport's K matrix."""
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

    return rotmat.quat_to_rotmat(quaternion.largest_eigenvector(k, xp=xp), xp=xp)
