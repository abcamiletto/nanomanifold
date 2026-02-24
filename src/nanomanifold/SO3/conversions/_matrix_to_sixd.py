"""Direct matrix -> 6D conversion."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace


def from_matrix_to_sixd(matrix: Float[Any, "... 3 3"], *, xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    if xp is None:
        xp = get_namespace(matrix)
    return xp.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)
