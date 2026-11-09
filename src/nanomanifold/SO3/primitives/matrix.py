"""Matrix edges: matrix <-> rotmat. A matrix is any 3x3 array; it becomes a rotation by projection onto SO(3)."""

from types import ModuleType
from typing import Any, Literal

from jaxtyping import Float

from . import quaternion, rotmat

ProjectionMode = Literal["svd", "davenport"]


def matrix_to_rotmat(matrix: Float[Any, "... 3 3"], *, mode: ProjectionMode = "svd", xp: ModuleType) -> Float[Any, "... 3 3"]:
    if mode == "svd":
        return _svd(matrix, xp)
    if mode == "davenport":
        return _davenport(matrix, xp)
    raise ValueError(f"Unsupported projection mode '{mode}'. Supported values: svd, davenport.")


def rotmat_to_matrix(rotmat: Float[Any, "... 3 3"], *, xp: ModuleType) -> Float[Any, "... 3 3"]:
    return rotmat


def _svd(matrix: Float[Any, "... 3 3"], xp: ModuleType) -> Float[Any, "... 3 3"]:
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


def _davenport(matrix: Float[Any, "... 3 3"], xp: ModuleType, *, steps: int = 15, eps: float = 1e-7) -> Float[Any, "... 3 3"]:
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
    identity = xp.stack(
        [
            xp.stack([one, zero, zero, zero], axis=-1),
            xp.stack([zero, one, zero, zero], axis=-1),
            xp.stack([zero, zero, one, zero], axis=-1),
            xp.stack([zero, zero, zero, one], axis=-1),
        ],
        axis=-2,
    )

    # Shift K so its largest algebraic eigenvalue also has the largest magnitude,
    # then square from the full eigenspace to avoid dependence on a single seed.
    projector = k + (xp.linalg.norm(k, axis=(-2, -1), keepdims=True) + eps) * identity
    for _ in range(steps):
        projector = projector / (xp.linalg.norm(projector, axis=(-2, -1), keepdims=True) + eps)
        projector = xp.matmul(projector, projector)

    diagonal = xp.stack([projector[..., i, i] for i in range(4)], axis=-1)
    column = xp.argmax(diagonal, axis=-1)
    seed = xp.stack([xp.where(column == i, one, zero) for i in range(4)], axis=-1)
    quat = xp.matmul(projector, seed[..., None])[..., 0]
    return rotmat.quat_to_rotmat(quaternion.canonicalize(quat, xp=xp), xp=xp)
