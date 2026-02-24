"""Direct Euler -> 6D conversion via direct matrix path."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from ._euler_to_matrix import from_euler_to_matrix
from ._matrix_to_sixd import from_matrix_to_sixd


def from_euler_to_sixd(euler: Float[Any, "... 3"], *, convention: str = "ZYX", xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    matrix = from_euler_to_matrix(euler, convention=convention, xp=xp)
    return from_matrix_to_sixd(matrix, xp=xp)
