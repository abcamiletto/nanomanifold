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

    src_euler_convention = "ZYX" if src_convention is None else src_convention
    dst_euler_convention = "ZYX" if dst_convention is None else dst_convention

    src_dispatch: str = src
    if src == "quat":
        if src_convention is None or src_convention == "wxyz":
            src_dispatch = "quat_wxyz"
        elif src_convention == "xyzw":
            src_dispatch = "quat_xyzw"
        else:
            raise ValueError(f"Unsupported quaternion convention '{src_convention}'.")

    dst_dispatch: str = dst
    if dst == "quat":
        if dst_convention is None or dst_convention == "wxyz":
            dst_dispatch = "quat_wxyz"
        elif dst_convention == "xyzw":
            dst_dispatch = "quat_xyzw"
        else:
            raise ValueError(f"Unsupported quaternion convention '{dst_convention}'.")

    if src_dispatch == dst_dispatch:
        if src_dispatch == "euler" and src_euler_convention != dst_euler_convention:
            return conversions.from_euler_to_euler(
                value,
                source_convention=src_euler_convention,
                target_convention=dst_euler_convention,
                xp=xp,
            )
        return value

    if src_dispatch == "axis_angle":
        if dst_dispatch == "euler":
            return conversions.from_axis_angle_to_euler(value, convention=dst_euler_convention, xp=xp)
        if dst_dispatch == "matrix" or dst_dispatch == "rotmat":
            return conversions.from_axis_angle_to_rotmat(value, xp=xp)
        if dst_dispatch == "quat_wxyz":
            return conversions.from_axis_angle_to_quat_wxyz(value, xp=xp)
        if dst_dispatch == "quat_xyzw":
            return conversions.from_axis_angle_to_quat_xyzw(value, xp=xp)
        if dst_dispatch == "sixd":
            return conversions.from_axis_angle_to_sixd(value, xp=xp)

    if src_dispatch == "euler":
        if dst_dispatch == "axis_angle":
            return conversions.from_euler_to_axis_angle(value, convention=src_euler_convention, xp=xp)
        if dst_dispatch == "matrix" or dst_dispatch == "rotmat":
            return conversions.from_euler_to_rotmat(value, convention=src_euler_convention, xp=xp)
        if dst_dispatch == "quat_wxyz":
            return conversions.from_euler_to_quat_wxyz(value, convention=src_euler_convention, xp=xp)
        if dst_dispatch == "quat_xyzw":
            return conversions.from_euler_to_quat_xyzw(value, convention=src_euler_convention, xp=xp)
        if dst_dispatch == "sixd":
            return conversions.from_euler_to_sixd(value, convention=src_euler_convention, xp=xp)

    if src_dispatch == "matrix":
        if dst_dispatch == "axis_angle":
            return conversions.from_matrix_to_axis_angle(value, xp=xp)
        if dst_dispatch == "euler":
            return conversions.from_matrix_to_euler(value, convention=dst_euler_convention, xp=xp)
        if dst_dispatch == "rotmat":
            return conversions.from_matrix_to_rotmat(value, xp=xp)
        if dst_dispatch == "quat_wxyz":
            return conversions.from_matrix_to_quat_wxyz(value, xp=xp)
        if dst_dispatch == "quat_xyzw":
            return conversions.from_matrix_to_quat_xyzw(value, xp=xp)
        if dst_dispatch == "sixd":
            return conversions.from_matrix_to_sixd(value, xp=xp)

    if src_dispatch == "rotmat":
        if dst_dispatch == "axis_angle":
            return conversions.from_rotmat_to_axis_angle(value, xp=xp)
        if dst_dispatch == "euler":
            return conversions.from_rotmat_to_euler(value, convention=dst_euler_convention, xp=xp)
        if dst_dispatch == "matrix":
            return value
        if dst_dispatch == "quat_wxyz":
            return conversions.from_rotmat_to_quat_wxyz(value, xp=xp)
        if dst_dispatch == "quat_xyzw":
            return conversions.from_rotmat_to_quat_xyzw(value, xp=xp)
        if dst_dispatch == "sixd":
            return conversions.from_rotmat_to_sixd(value, xp=xp)

    if src_dispatch == "quat_wxyz":
        if dst_dispatch == "axis_angle":
            return conversions.from_quat_wxyz_to_axis_angle(value, xp=xp)
        if dst_dispatch == "euler":
            return conversions.from_quat_wxyz_to_euler(value, convention=dst_euler_convention, xp=xp)
        if dst_dispatch == "matrix" or dst_dispatch == "rotmat":
            return conversions.from_quat_wxyz_to_rotmat(value, xp=xp)
        if dst_dispatch == "quat_xyzw":
            return conversions.from_quat_wxyz_to_quat_xyzw(value, xp=xp)
        if dst_dispatch == "sixd":
            return conversions.from_quat_wxyz_to_sixd(value, xp=xp)

    if src_dispatch == "quat_xyzw":
        if dst_dispatch == "axis_angle":
            return conversions.from_quat_xyzw_to_axis_angle(value, xp=xp)
        if dst_dispatch == "euler":
            return conversions.from_quat_xyzw_to_euler(value, convention=dst_euler_convention, xp=xp)
        if dst_dispatch == "matrix" or dst_dispatch == "rotmat":
            return conversions.from_quat_xyzw_to_rotmat(value, xp=xp)
        if dst_dispatch == "quat_wxyz":
            return conversions.from_quat_xyzw_to_quat_wxyz(value, xp=xp)
        if dst_dispatch == "sixd":
            return conversions.from_quat_xyzw_to_sixd(value, xp=xp)

    if src_dispatch == "sixd":
        if dst_dispatch == "axis_angle":
            return conversions.from_sixd_to_axis_angle(value, xp=xp)
        if dst_dispatch == "euler":
            return conversions.from_sixd_to_euler(value, convention=dst_euler_convention, xp=xp)
        if dst_dispatch == "matrix" or dst_dispatch == "rotmat":
            return conversions.from_sixd_to_rotmat(value, xp=xp)
        if dst_dispatch == "quat_wxyz":
            return conversions.from_sixd_to_quat_wxyz(value, xp=xp)
        if dst_dispatch == "quat_xyzw":
            return conversions.from_sixd_to_quat_xyzw(value, xp=xp)

    raise ValueError(f"Unsupported conversion from '{src_dispatch}' to '{dst_dispatch}'.")


__all__ = ["RotationRep", "convert"]
