"""Direct 6D -> matrix conversion."""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold.common import get_namespace

from ..primitives.sixd import _from_6d_to_matrix


def from_sixd_to_matrix(sixd: Float[Any, "... 6"], *, xp: ModuleType | None = None) -> Float[Any, "... 3 3"]:
    if xp is None:
        xp = get_namespace(sixd)
    return _from_6d_to_matrix(sixd, xp)
