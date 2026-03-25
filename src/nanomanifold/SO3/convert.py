"""Convenience dispatcher for SO(3) representation conversions."""

from types import ModuleType
from typing import Any, Literal

from jaxtyping import Float

from . import conversions

RotationRep = Literal["axis_angle", "euler", "matrix", "rotmat", "quat", "sixd"]

_REPRESENTATIONS = ("axis_angle", "euler", "matrix", "rotmat", "quat", "sixd")


def convert(
    value: Float[Any, "..."],
    *,
    src: RotationRep,
    dst: RotationRep,
    src_convention: str | None = None,
    dst_convention: str | None = None,
    xp: ModuleType | None = None,
) -> Float[Any, "..."]:
    """Convert between SO(3) representations selected at runtime."""
    if src not in _REPRESENTATIONS:
        supported = ", ".join(_REPRESENTATIONS)
        raise ValueError(f"Unsupported rotation representation '{src}'. Supported values: {supported}.")
    if dst not in _REPRESENTATIONS:
        supported = ", ".join(_REPRESENTATIONS)
        raise ValueError(f"Unsupported rotation representation '{dst}'. Supported values: {supported}.")

    src_convention = "wxyz" if src == "quat" and src_convention is None else "ZYX" if src_convention is None else src_convention
    dst_convention = "wxyz" if dst == "quat" and dst_convention is None else "ZYX" if dst_convention is None else dst_convention

    if src == dst:
        if src == "euler" and src_convention != dst_convention:
            return conversions.from_euler_to_euler(
                value,
                src_convention=src_convention,
                dst_convention=dst_convention,
                xp=xp,
            )
        if src == "quat" and src_convention != dst_convention:
            return conversions.from_quat_to_quat(
                value,
                src_convention=src_convention,
                dst_convention=dst_convention,
                xp=xp,
            )
        return value

    if src == "axis_angle":
        if dst == "euler":
            return conversions.from_axis_angle_to_euler(value, convention=dst_convention, xp=xp)
        if dst == "rotmat" or dst == "matrix":
            return conversions.from_axis_angle_to_rotmat(value, xp=xp)
        if dst == "quat":
            return conversions.from_axis_angle_to_quat(value, convention=dst_convention, xp=xp)
        if dst == "sixd":
            return conversions.from_axis_angle_to_sixd(value, xp=xp)

    if src == "euler":
        if dst == "axis_angle":
            return conversions.from_euler_to_axis_angle(value, convention=src_convention, xp=xp)
        if dst == "rotmat" or dst == "matrix":
            return conversions.from_euler_to_rotmat(value, convention=src_convention, xp=xp)
        if dst == "quat":
            return conversions.from_euler_to_quat(
                value,
                src_convention=src_convention,
                dst_convention=dst_convention,
                xp=xp,
            )
        if dst == "sixd":
            return conversions.from_euler_to_sixd(value, convention=src_convention, xp=xp)

    if src == "matrix":
        if dst == "axis_angle":
            return conversions.from_matrix_to_axis_angle(value, xp=xp)
        if dst == "euler":
            return conversions.from_matrix_to_euler(value, convention=dst_convention, xp=xp)
        if dst == "rotmat":
            return conversions.from_matrix_to_rotmat(value, xp=xp)
        if dst == "quat":
            return conversions.from_matrix_to_quat(value, convention=dst_convention, xp=xp)
        if dst == "sixd":
            return conversions.from_matrix_to_sixd(value, xp=xp)

    if src == "rotmat":
        if dst == "axis_angle":
            return conversions.from_rotmat_to_axis_angle(value, xp=xp)
        if dst == "euler":
            return conversions.from_rotmat_to_euler(value, convention=dst_convention, xp=xp)
        if dst == "matrix":
            return value
        if dst == "quat":
            return conversions.from_rotmat_to_quat(value, convention=dst_convention, xp=xp)
        if dst == "sixd":
            return conversions.from_rotmat_to_sixd(value, xp=xp)

    if src == "quat":
        if dst == "axis_angle":
            return conversions.from_quat_to_axis_angle(value, convention=src_convention, xp=xp)
        if dst == "euler":
            return conversions.from_quat_to_euler(
                value,
                src_convention=src_convention,
                dst_convention=dst_convention,
                xp=xp,
            )
        if dst == "rotmat" or dst == "matrix":
            return conversions.from_quat_to_rotmat(value, convention=src_convention, xp=xp)
        if dst == "sixd":
            return conversions.from_quat_to_sixd(value, convention=src_convention, xp=xp)

    if src == "sixd":
        if dst == "axis_angle":
            return conversions.from_sixd_to_axis_angle(value, xp=xp)
        if dst == "euler":
            return conversions.from_sixd_to_euler(value, convention=dst_convention, xp=xp)
        if dst == "rotmat" or dst == "matrix":
            return conversions.from_sixd_to_rotmat(value, xp=xp)
        if dst == "quat":
            return conversions.from_sixd_to_quat(value, convention=dst_convention, xp=xp)

    raise ValueError(f"Unsupported conversion from '{src}' to '{dst}'.")


__all__ = ["RotationRep", "convert"]
