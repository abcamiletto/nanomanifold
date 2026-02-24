"""Direct Euler -> matrix conversion."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from ..primitives.euler import _euler_to_matrix


def from_euler_to_matrix(euler: Float[Any, "... 3"], *, convention: str = "ZYX", xp: ModuleType | None = None) -> Float[Any, "... 3 3"]:
    return _euler_to_matrix(euler, convention, xp=xp)
