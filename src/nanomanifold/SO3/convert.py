"""Convenience dispatcher for SO(3) representation conversions."""

from types import ModuleType
from typing import Any, Literal

from jaxtyping import Float

from . import conversions

RotationRep = Literal["axis_angle", "euler", "matrix", "rotmat", "quat_wxyz", "quat_xyzw", "sixd"]

_REPRESENTATIONS = ("axis_angle", "euler", "matrix", "rotmat", "quat_wxyz", "quat_xyzw", "sixd")


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

    src_euler_convention = "ZYX" if src_convention is None else src_convention
    dst_euler_convention = "ZYX" if dst_convention is None else dst_convention

    if src == dst:
        if src == "euler" and src_euler_convention != dst_euler_convention:
            return conversions.from_euler_to_euler(
                value,
                source_convention=src_euler_convention,
                target_convention=dst_euler_convention,
                xp=xp,
            )
        return value

    if src == "axis_angle":
        if dst == "euler":
            return conversions.from_axis_angle_to_euler(value, convention=dst_euler_convention, xp=xp)
        if dst == "matrix" or dst == "rotmat":
            return conversions.from_axis_angle_to_rotmat(value, xp=xp)
        if dst == "quat_wxyz":
            return conversions.from_axis_angle_to_quat_wxyz(value, xp=xp)
        if dst == "quat_xyzw":
            return conversions.from_axis_angle_to_quat_xyzw(value, xp=xp)
        if dst == "sixd":
            return conversions.from_axis_angle_to_sixd(value, xp=xp)

    if src == "euler":
        if dst == "axis_angle":
            return conversions.from_euler_to_axis_angle(value, convention=src_euler_convention, xp=xp)
        if dst == "matrix" or dst == "rotmat":
            return conversions.from_euler_to_rotmat(value, convention=src_euler_convention, xp=xp)
        if dst == "quat_wxyz":
            return conversions.from_euler_to_quat_wxyz(value, convention=src_euler_convention, xp=xp)
        if dst == "quat_xyzw":
            return conversions.from_euler_to_quat_xyzw(value, convention=src_euler_convention, xp=xp)
        if dst == "sixd":
            return conversions.from_euler_to_sixd(value, convention=src_euler_convention, xp=xp)

    if src == "matrix":
        if dst == "axis_angle":
            return conversions.from_matrix_to_axis_angle(value, xp=xp)
        if dst == "euler":
            return conversions.from_matrix_to_euler(value, convention=dst_euler_convention, xp=xp)
        if dst == "rotmat":
            return conversions.from_matrix_to_rotmat(value, xp=xp)
        if dst == "quat_wxyz":
            return conversions.from_matrix_to_quat_wxyz(value, xp=xp)
        if dst == "quat_xyzw":
            return conversions.from_matrix_to_quat_xyzw(value, xp=xp)
        if dst == "sixd":
            return conversions.from_matrix_to_sixd(value, xp=xp)

    if src == "rotmat":
        if dst == "axis_angle":
            return conversions.from_rotmat_to_axis_angle(value, xp=xp)
        if dst == "euler":
            return conversions.from_rotmat_to_euler(value, convention=dst_euler_convention, xp=xp)
        if dst == "matrix":
            return value
        if dst == "quat_wxyz":
            return conversions.from_rotmat_to_quat_wxyz(value, xp=xp)
        if dst == "quat_xyzw":
            return conversions.from_rotmat_to_quat_xyzw(value, xp=xp)
        if dst == "sixd":
            return conversions.from_rotmat_to_sixd(value, xp=xp)

    if src == "quat_wxyz":
        if dst == "axis_angle":
            return conversions.from_quat_wxyz_to_axis_angle(value, xp=xp)
        if dst == "euler":
            return conversions.from_quat_wxyz_to_euler(value, convention=dst_euler_convention, xp=xp)
        if dst == "matrix" or dst == "rotmat":
            return conversions.from_quat_wxyz_to_rotmat(value, xp=xp)
        if dst == "quat_xyzw":
            return conversions.from_quat_wxyz_to_quat_xyzw(value, xp=xp)
        if dst == "sixd":
            return conversions.from_quat_wxyz_to_sixd(value, xp=xp)

    if src == "quat_xyzw":
        if dst == "axis_angle":
            return conversions.from_quat_xyzw_to_axis_angle(value, xp=xp)
        if dst == "euler":
            return conversions.from_quat_xyzw_to_euler(value, convention=dst_euler_convention, xp=xp)
        if dst == "matrix" or dst == "rotmat":
            return conversions.from_quat_xyzw_to_rotmat(value, xp=xp)
        if dst == "quat_wxyz":
            return conversions.from_quat_xyzw_to_quat_wxyz(value, xp=xp)
        if dst == "sixd":
            return conversions.from_quat_xyzw_to_sixd(value, xp=xp)

    if src == "sixd":
        if dst == "axis_angle":
            return conversions.from_sixd_to_axis_angle(value, xp=xp)
        if dst == "euler":
            return conversions.from_sixd_to_euler(value, convention=dst_euler_convention, xp=xp)
        if dst == "matrix" or dst == "rotmat":
            return conversions.from_sixd_to_rotmat(value, xp=xp)
        if dst == "quat_wxyz":
            return conversions.from_sixd_to_quat_wxyz(value, xp=xp)
        if dst == "quat_xyzw":
            return conversions.from_sixd_to_quat_xyzw(value, xp=xp)

    raise ValueError(f"Unsupported conversion from '{src}' to '{dst}'.")


__all__ = ["RotationRep", "convert"]
