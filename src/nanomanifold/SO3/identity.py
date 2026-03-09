from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import common
from nanomanifold.common import get_namespace

from .convert import RotationRep


def identity_as(
    ref: Float[Any, "..."],
    *,
    rotation_type: RotationRep = "quat",
    xp: ModuleType | None = None,
) -> Float[Any, "..."]:
    """Return an identity rotation matching ref backend, dtype, and batch dimensions."""
    if xp is None:
        xp = get_namespace(ref)

    if ref.shape[-2:] == (3, 3):
        batch_dims = ref.shape[:-2]
    else:
        batch_dims = ref.shape[:-1]

    if rotation_type == "matrix":
        if "torch" in xp.__name__:
            eye = xp.eye(3, dtype=ref.dtype, device=ref.device)
        else:
            eye = xp.eye(3, dtype=ref.dtype)
        return xp.broadcast_to(eye, batch_dims + (3, 3))

    if rotation_type == "axis_angle" or rotation_type == "euler":
        return common.zeros_as(ref, shape=batch_dims + (3,), xp=xp)

    if rotation_type == "quat":
        zeros = common.zeros_as(ref, shape=batch_dims + (4,), xp=xp)
        one = zeros[..., :1] + 1
        return xp.concatenate([one, zeros[..., 1:]], axis=-1)

    if rotation_type == "sixd":
        zeros = common.zeros_as(ref, shape=batch_dims + (6,), xp=xp)
        zero = zeros[..., :1]
        one = zero + 1
        return xp.concatenate([one, zero, zero, zero, one, zero], axis=-1)

    raise ValueError(f"Unsupported rotation_type '{rotation_type}'.")


__all__ = ["identity_as"]
