"""Conversions between SO(3) representations.

Representations form a tree rooted at the canonical quaternion (unit norm, wxyz order, w >= 0):

    canonical quaternion
    ├── quat           convention
    ├── euler          convention
    ├── axis_angle
    │   └── hinge      axes
    └── rotmat
        ├── sixd
        └── matrix     mode

Each representation only defines the edge to its parent and back. A conversion climbs from
``src`` to the closest common ancestor with ``dst`` and descends to ``dst``, so every pair has
exactly one route made of representation-native steps (e.g. sixd -> rotmat never touches a
quaternion). Routes are resolved from static strings, so tracing compilers only see array ops.
"""

from collections.abc import Callable
from types import ModuleType
from typing import Any, Literal, NamedTuple

from jaxtyping import Float

from nanomanifold.common import get_namespace

from .primitives import axis_angle, euler, hinge, matrix, quaternion, rotmat, sixd
from .primitives.euler import EulerConvention
from .primitives.matrix import ProjectionMode
from .primitives.quaternion import QuaternionConvention

RotationRep = Literal["axis_angle", "euler", "hinge", "matrix", "rotmat", "quat", "sixd"]


def convert(
    value: Float[Any, "..."],
    *,
    src: RotationRep,
    dst: RotationRep,
    src_kwargs: dict[str, Any] = {},
    dst_kwargs: dict[str, Any] = {},
    xp: ModuleType | None = None,
) -> Float[Any, "..."]:
    """Convert ``value`` from the ``src`` to the ``dst`` representation.

    ``src_kwargs`` and ``dst_kwargs`` configure the source and target representations, e.g.
    ``{"convention": "xyzw"}`` for ``quat``, ``{"convention": "XYZ"}`` for ``euler``,
    ``{"mode": "davenport"}`` for ``matrix`` or ``{"axes": axes}`` for ``hinge``.
    """
    if xp is None:
        xp = get_namespace(value)
    climb, descend = _ROUTES[src, dst]
    for rep in climb:
        value = _TREE[rep].to_parent(value, **(src_kwargs if rep == src else {}), xp=xp)
    for rep in descend:
        value = _TREE[rep].from_parent(value, **(dst_kwargs if rep == dst else {}), xp=xp)
    return value


def from_axis_angle(
    axis_angle: Float[Any, "... 3"], *, convention: QuaternionConvention = "wxyz", xp: ModuleType | None = None
) -> Float[Any, "... 4"]:
    return convert(axis_angle, src="axis_angle", dst="quat", dst_kwargs={"convention": convention}, xp=xp)


def to_axis_angle(
    q: Float[Any, "... 4"], *, convention: QuaternionConvention = "wxyz", xp: ModuleType | None = None
) -> Float[Any, "... 3"]:
    return convert(q, src="quat", dst="axis_angle", src_kwargs={"convention": convention}, xp=xp)


def from_hinge(
    angles: Float[Any, "... 1"],
    axes: Float[Any, "... 3"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    return convert(angles, src="hinge", dst="quat", src_kwargs={"axes": axes}, dst_kwargs={"convention": convention}, xp=xp)


def to_hinge(
    q: Float[Any, "... 4"],
    axes: Float[Any, "... 3"],
    *,
    convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 1"]:
    return convert(q, src="quat", dst="hinge", src_kwargs={"convention": convention}, dst_kwargs={"axes": axes}, xp=xp)


def from_euler(
    euler: Float[Any, "... 3"],
    *,
    convention: EulerConvention = "ZYX",
    quat_convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    return convert(euler, src="euler", dst="quat", src_kwargs={"convention": convention}, dst_kwargs={"convention": quat_convention}, xp=xp)


def to_euler(
    q: Float[Any, "... 4"],
    *,
    convention: EulerConvention = "ZYX",
    quat_convention: QuaternionConvention = "wxyz",
    xp: ModuleType | None = None,
) -> Float[Any, "... 3"]:
    return convert(q, src="quat", dst="euler", src_kwargs={"convention": quat_convention}, dst_kwargs={"convention": convention}, xp=xp)


def from_rotmat(
    rotmat: Float[Any, "... 3 3"], *, convention: QuaternionConvention = "wxyz", xp: ModuleType | None = None
) -> Float[Any, "... 4"]:
    return convert(rotmat, src="rotmat", dst="quat", dst_kwargs={"convention": convention}, xp=xp)


def to_rotmat(q: Float[Any, "... 4"], *, convention: QuaternionConvention = "wxyz", xp: ModuleType | None = None) -> Float[Any, "... 3 3"]:
    return convert(q, src="quat", dst="rotmat", src_kwargs={"convention": convention}, xp=xp)


def from_matrix(
    matrix: Float[Any, "... 3 3"],
    *,
    convention: QuaternionConvention = "wxyz",
    mode: ProjectionMode = "svd",
    xp: ModuleType | None = None,
) -> Float[Any, "... 4"]:
    return convert(matrix, src="matrix", dst="quat", src_kwargs={"mode": mode}, dst_kwargs={"convention": convention}, xp=xp)


def from_sixd(sixd: Float[Any, "... 6"], *, convention: QuaternionConvention = "wxyz", xp: ModuleType | None = None) -> Float[Any, "... 4"]:
    return convert(sixd, src="sixd", dst="quat", dst_kwargs={"convention": convention}, xp=xp)


def to_sixd(q: Float[Any, "... 4"], *, convention: QuaternionConvention = "wxyz", xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    return convert(q, src="quat", dst="sixd", src_kwargs={"convention": convention}, xp=xp)


class _Node(NamedTuple):
    parent: str
    to_parent: Callable[..., Any]
    from_parent: Callable[..., Any]


_ROOT = "canonical_quat"

_TREE: dict[str, _Node] = {
    "quat": _Node(_ROOT, quaternion.from_quat, quaternion.to_quat),
    "axis_angle": _Node(_ROOT, axis_angle.axis_angle_to_quat, axis_angle.quat_to_axis_angle),
    "rotmat": _Node(_ROOT, rotmat.rotmat_to_quat, rotmat.quat_to_rotmat),
    "hinge": _Node("axis_angle", hinge.hinge_to_axis_angle, hinge.axis_angle_to_hinge),
    "sixd": _Node("rotmat", sixd.sixd_to_rotmat, sixd.rotmat_to_sixd),
    "euler": _Node(_ROOT, euler.euler_to_quat, euler.quat_to_euler),
    "matrix": _Node("rotmat", matrix.matrix_to_rotmat, matrix.rotmat_to_matrix),
}


def _route(src: str, dst: str) -> tuple[list[str], list[str]]:
    """Representations to climb out of from ``src`` and to descend into towards ``dst``."""
    src_chain = _ancestry(src)
    dst_chain = _ancestry(dst)
    # Converting a representation to itself round-trips through its parent, which applies the kwargs change.
    meet = _TREE[src].parent if src == dst else next(rep for rep in src_chain if rep in dst_chain)
    climb = src_chain[: src_chain.index(meet)]
    descend = dst_chain[: dst_chain.index(meet)][::-1]
    return climb, descend


def _ancestry(rep: str) -> list[str]:
    chain = [rep]
    while chain[-1] != _ROOT:
        chain.append(_TREE[chain[-1]].parent)
    return chain


_ROUTES = {(src, dst): _route(src, dst) for src in _TREE for dst in _TREE}


__all__ = [
    "RotationRep",
    "convert",
    "from_axis_angle",
    "to_axis_angle",
    "from_hinge",
    "to_hinge",
    "from_euler",
    "to_euler",
    "from_rotmat",
    "to_rotmat",
    "from_matrix",
    "from_sixd",
    "to_sixd",
]
