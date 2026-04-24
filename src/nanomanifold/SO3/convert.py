"""Convenience dispatcher for SO(3) representation conversions."""

from types import ModuleType
from typing import Any, Literal

from jaxtyping import Float

from . import conversions, hinge
from .primitives import axis_angle, euler, quaternion, rotmat, sixd

RotationRep = Literal["axis_angle", "euler", "hinge", "matrix", "rotmat", "quat", "sixd"]
RotationSourceRep = Literal["axis_angle", "euler", "hinge", "matrix", "rotmat", "quat", "sixd"]


def convert(
    value: Float[Any, "..."],
    *,
    src: RotationSourceRep,
    dst: RotationRep,
    src_kwargs: dict[str, Any] = {},
    dst_kwargs: dict[str, Any] = {},
    xp: ModuleType | None = None,
) -> Float[Any, "..."]:
    """Convert between SO(3) representations selected at runtime."""
    dst = "rotmat" if dst == "matrix" else dst

    if src == dst and not src_kwargs and not dst_kwargs:
        return value

    if src == "hinge" and dst != "hinge":
        return getattr(conversions, f"from_hinge_to_{dst}")(value, **src_kwargs, **dst_kwargs, xp=xp)

    if dst == "hinge" and src != "hinge":
        return getattr(conversions, f"from_{src}_to_hinge")(value, **src_kwargs, **dst_kwargs, xp=xp)

    quat = _to_canonical_quat(value, src, src_kwargs, xp=xp)
    return _from_canonical_quat(quat, dst, dst_kwargs, xp=xp)


def _to_canonical_quat(
    value: Float[Any, "..."],
    src: RotationSourceRep,
    kwargs: dict[str, Any],
    *,
    xp: ModuleType | None,
) -> Float[Any, "... 4"]:
    if src == "axis_angle":
        return axis_angle.from_axis_angle(value, **kwargs, xp=xp)
    if src == "euler":
        return euler.from_euler(value, **kwargs, xp=xp)
    if src == "hinge":
        return hinge.from_hinge(value, **kwargs, xp=xp)
    if src == "matrix":
        return rotmat.from_matrix(value, **kwargs, xp=xp)
    if src == "rotmat":
        return rotmat.from_rotmat(value, **kwargs, xp=xp)
    if src == "quat":
        return quaternion.from_quat(value, **kwargs, xp=xp)
    if src == "sixd":
        return sixd.from_sixd(value, **kwargs, xp=xp)
    raise ValueError(f"Unsupported source representation '{src}'.")


def _from_canonical_quat(
    quat: Float[Any, "... 4"],
    dst: RotationRep,
    kwargs: dict[str, Any],
    *,
    xp: ModuleType | None,
) -> Float[Any, "..."]:
    if dst == "axis_angle":
        return axis_angle.to_axis_angle(quat, **kwargs, xp=xp)
    if dst == "euler":
        return euler.to_euler(quat, **kwargs, xp=xp)
    if dst == "hinge":
        return hinge.to_hinge(quat, **kwargs, xp=xp)
    if dst == "rotmat":
        return rotmat.to_rotmat(quat, **kwargs, xp=xp)
    if dst == "quat":
        return quaternion.to_quat(quat, **kwargs, xp=xp)
    if dst == "sixd":
        return sixd.to_sixd(quat, **kwargs, xp=xp)
    raise ValueError(f"Unsupported target representation '{dst}'.")


__all__ = ["RotationRep", "RotationSourceRep", "convert"]
