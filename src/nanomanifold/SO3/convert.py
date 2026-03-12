"""Convenience dispatcher for SO(3) representation conversions."""

from types import ModuleType
from typing import Any, Literal

from jaxtyping import Float

from . import conversions

RotationRep = Literal["axis_angle", "euler", "matrix", "quat", "sixd"]
DispatchRotationRep = Literal["axis_angle", "euler", "matrix", "quat_wxyz", "quat_xyzw", "sixd"]

_REPRESENTATIONS = ("axis_angle", "euler", "matrix", "quat", "sixd")


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

    src_rep: DispatchRotationRep
    if src == "quat":
        if src_convention is None or src_convention == "wxyz":
            src_rep = "quat_wxyz"
        elif src_convention == "xyzw":
            src_rep = "quat_xyzw"
        else:
            raise ValueError(f"Unsupported quaternion convention '{src_convention}'.")
    else:
        src_rep = src

    dst_rep: DispatchRotationRep
    if dst == "quat":
        if dst_convention is None or dst_convention == "wxyz":
            dst_rep = "quat_wxyz"
        elif dst_convention == "xyzw":
            dst_rep = "quat_xyzw"
        else:
            raise ValueError(f"Unsupported quaternion convention '{dst_convention}'.")
    else:
        dst_rep = dst

    if src_rep == dst_rep:
        if src_rep == "euler" and src_euler_convention != dst_euler_convention:
            return conversions.from_euler_to_euler(
                value,
                source_convention=src_euler_convention,
                target_convention=dst_euler_convention,
                xp=xp,
            )
        return value

    if src_rep == "axis_angle":
        if dst_rep == "euler":
            return conversions.from_axis_angle_to_euler(value, convention=dst_euler_convention, xp=xp)
        if dst_rep == "matrix":
            return conversions.from_axis_angle_to_matrix(value, xp=xp)
        if dst_rep == "quat_wxyz":
            return conversions.from_axis_angle_to_quat_wxyz(value, xp=xp)
        if dst_rep == "quat_xyzw":
            return conversions.from_axis_angle_to_quat_xyzw(value, xp=xp)
        if dst_rep == "sixd":
            return conversions.from_axis_angle_to_sixd(value, xp=xp)

    if src_rep == "euler":
        if dst_rep == "axis_angle":
            return conversions.from_euler_to_axis_angle(value, convention=src_euler_convention, xp=xp)
        if dst_rep == "matrix":
            return conversions.from_euler_to_matrix(value, convention=src_euler_convention, xp=xp)
        if dst_rep == "quat_wxyz":
            return conversions.from_euler_to_quat_wxyz(value, convention=src_euler_convention, xp=xp)
        if dst_rep == "quat_xyzw":
            return conversions.from_euler_to_quat_xyzw(value, convention=src_euler_convention, xp=xp)
        if dst_rep == "sixd":
            return conversions.from_euler_to_sixd(value, convention=src_euler_convention, xp=xp)

    if src_rep == "matrix":
        if dst_rep == "axis_angle":
            return conversions.from_matrix_to_axis_angle(value, xp=xp)
        if dst_rep == "euler":
            return conversions.from_matrix_to_euler(value, convention=dst_euler_convention, xp=xp)
        if dst_rep == "quat_wxyz":
            return conversions.from_matrix_to_quat_wxyz(value, xp=xp)
        if dst_rep == "quat_xyzw":
            return conversions.from_matrix_to_quat_xyzw(value, xp=xp)
        if dst_rep == "sixd":
            return conversions.from_matrix_to_sixd(value, xp=xp)

    if src_rep == "quat_wxyz":
        if dst_rep == "axis_angle":
            return conversions.from_quat_wxyz_to_axis_angle(value, xp=xp)
        if dst_rep == "euler":
            return conversions.from_quat_wxyz_to_euler(value, convention=dst_euler_convention, xp=xp)
        if dst_rep == "matrix":
            return conversions.from_quat_wxyz_to_matrix(value, xp=xp)
        if dst_rep == "quat_xyzw":
            return conversions.from_quat_wxyz_to_quat_xyzw(value, xp=xp)
        if dst_rep == "sixd":
            return conversions.from_quat_wxyz_to_sixd(value, xp=xp)

    if src_rep == "quat_xyzw":
        if dst_rep == "axis_angle":
            return conversions.from_quat_xyzw_to_axis_angle(value, xp=xp)
        if dst_rep == "euler":
            return conversions.from_quat_xyzw_to_euler(value, convention=dst_euler_convention, xp=xp)
        if dst_rep == "matrix":
            return conversions.from_quat_xyzw_to_matrix(value, xp=xp)
        if dst_rep == "quat_wxyz":
            return conversions.from_quat_xyzw_to_quat_wxyz(value, xp=xp)
        if dst_rep == "sixd":
            return conversions.from_quat_xyzw_to_sixd(value, xp=xp)

    if src_rep == "sixd":
        if dst_rep == "axis_angle":
            return conversions.from_sixd_to_axis_angle(value, xp=xp)
        if dst_rep == "euler":
            return conversions.from_sixd_to_euler(value, convention=dst_euler_convention, xp=xp)
        if dst_rep == "matrix":
            return conversions.from_sixd_to_matrix(value, xp=xp)
        if dst_rep == "quat_wxyz":
            return conversions.from_sixd_to_quat_wxyz(value, xp=xp)
        if dst_rep == "quat_xyzw":
            return conversions.from_sixd_to_quat_xyzw(value, xp=xp)

    raise ValueError(f"Unsupported conversion from '{src_rep}' to '{dst_rep}'.")


__all__ = ["DispatchRotationRep", "RotationRep", "convert"]
