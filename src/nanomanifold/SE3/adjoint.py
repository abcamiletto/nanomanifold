r"""Adjoint representation for :math:`\mathrm{SE}(3)` transformations."""

from typing import Any

from jaxtyping import Float

from ..common import get_namespace
from ..SO3 import hat
from ..SO3.conversions.matrix import to_matrix
from .canonicalize import canonicalize


def adjoint(se3: Float[Any, "... 7"]) -> Float[Any, "... 6 6"]:
    r"""Compute the adjoint matrix of an :math:`\mathrm{SE}(3)` element."""

    xp = get_namespace(se3)

    se3 = canonicalize(se3)

    rotation = to_matrix(se3[..., :4])
    translation = se3[..., 4:7]

    translation_hat = hat(translation)
    translation_term = xp.matmul(translation_hat, rotation)

    zeros = rotation * 0.0

    top = xp.concatenate([rotation, zeros], axis=-1)
    bottom = xp.concatenate([translation_term, rotation], axis=-1)

    return xp.concatenate([top, bottom], axis=-2)
