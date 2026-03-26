from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace

from .primitives.quaternion import QuaternionConvention, canonicalize, from_quat, to_quat


def inverse(
    q: Float[Any, "... 4"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    assert convention in ("wxyz", "xyzw"), "Quaternion convention must be 'wxyz' or 'xyzw'."
    if xp is None:
        xp = get_namespace(q)
    q = from_quat(q, convention=convention, xp=xp)
    q = canonicalize(q, xp=xp)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    q_inv = xp.stack([w, -x, -y, -z], axis=-1)

    return to_quat(canonicalize(q_inv, xp=xp), convention=convention, xp=xp)
