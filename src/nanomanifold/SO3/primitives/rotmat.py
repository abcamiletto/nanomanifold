"""Rotation-matrix edges: rotmat <-> canonical quaternion."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from . import quaternion


def rotmat_to_quat(rotmat: Float[Any, "... 3 3"], *, xp: ModuleType) -> Float[Any, "... 4"]:
    """Shepperd's method: normalize the best-conditioned row of ``4 q qᵀ``, the one whose diagonal entry is largest.

    Each row is a multiple of the quaternion, so normalizing it replaces the usual square root and divisions.
    """
    # Each entry is read once: autodiff scatters every read back into a full-size array.
    m00, m01, m02 = rotmat[..., 0, 0], rotmat[..., 0, 1], rotmat[..., 0, 2]
    m10, m11, m12 = rotmat[..., 1, 0], rotmat[..., 1, 1], rotmat[..., 1, 2]
    m20, m21, m22 = rotmat[..., 2, 0], rotmat[..., 2, 1], rotmat[..., 2, 2]
    trace = m00 + m11 + m22
    one = xp.ones_like(trace)  # not the literal 1: NumPy 1.x promotes scalar + Python number to float64
    wx, wy, wz = m21 - m12, m02 - m20, m10 - m01
    xy, xz, yz = m01 + m10, m02 + m20, m12 + m21

    use_w = trace > 0
    use_x = (m00 > m11) & (m00 > m22)
    use_y = m11 > m22

    def select(row_w, row_x, row_y, row_z):
        return xp.where(use_w, row_w, xp.where(use_x, row_x, xp.where(use_y, row_y, row_z)))

    w = select(one + trace, wx, wy, wz)
    x = select(wx, one + m00 - m11 - m22, xy, xz)
    y = select(wy, xy, one - m00 + m11 - m22, yz)
    z = select(wz, xz, yz, one - m00 - m11 + m22)
    return quaternion.normalize(xp.stack([w, x, y, z], axis=-1), xp)


def quat_to_rotmat(q: Float[Any, "... 4"], *, xp: ModuleType) -> Float[Any, "... 3 3"]:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # Each product once, doubled through one factor: doubling is exact, so this rounds like 2 (x y - w z).
    x2, y2, z2 = x + x, y + y, z + z
    xx, yy, zz = x * x2, y * y2, z * z2
    xy, xz, yz = x * y2, x * z2, y * z2
    wx, wy, wz = w * x2, w * y2, w * z2
    one = xp.ones_like(w)

    return xp.stack(
        [
            xp.stack([one - (yy + zz), xy - wz, xz + wy], axis=-1),
            xp.stack([xy + wz, one - (xx + zz), yz - wx], axis=-1),
            xp.stack([xz - wy, yz + wx, one - (xx + yy)], axis=-1),
        ],
        axis=-2,
    )
