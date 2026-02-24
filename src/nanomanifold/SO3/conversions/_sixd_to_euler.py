"""Direct 6D -> Euler conversion via direct matrix path."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from ._matrix_to_euler import from_matrix_to_euler
from ._sixd_to_matrix import from_sixd_to_matrix


def from_sixd_to_euler(sixd: Float[Any, "... 6"], *, convention: str = "ZYX", xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    matrix = from_sixd_to_matrix(sixd, xp=xp)
    return from_matrix_to_euler(matrix, convention=convention, xp=xp)
