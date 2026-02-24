"""Direct matrix -> Euler conversion."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from ..primitives.euler import _matrix_to_euler


def from_matrix_to_euler(matrix: Float[Any, "... 3 3"], *, convention: str = "ZYX", xp: ModuleType | None = None) -> Float[Any, "... 3"]:
    return _matrix_to_euler(matrix, convention, xp=xp)
