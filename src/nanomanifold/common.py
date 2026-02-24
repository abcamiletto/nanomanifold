import math
from types import ModuleType

import array_api_compat


def get_namespace(array) -> ModuleType:
    return array_api_compat.get_namespace(array)


def get_namespace_by_name(name: str) -> ModuleType:
    if name == "numpy":
        import numpy as np

        ones = np.ones(1)
        return get_namespace(ones)

    elif name == "torch":
        import torch

        ones = torch.ones(1)
        return get_namespace(ones)

    elif name == "jax":
        import jax.numpy as jnp

        ones = jnp.ones(1)
        return get_namespace(ones)

    else:
        raise ValueError(f"Unknown array namespace '{name}'. Supported: 'numpy', 'torch', 'jax'.")


def safe_eps(dtype, xp, scale=10.0) -> float:
    """Machine epsilon * scale. Used for division-by-zero guards."""
    return float(xp.finfo(dtype).eps) * scale


def small_angle_threshold(dtype, xp) -> float:
    """Dtype-dependent threshold for small-angle approximations.
    Uses sqrt(eps): ~0.031 for f16, ~3.5e-4 for f32, ~1.5e-8 for f64.
    At theta ~ sqrt(eps), Taylor series error is O(eps)."""
    return math.sqrt(safe_eps(dtype, xp, scale=1.0))


def slerp_linear_threshold(dtype, xp) -> float:
    """Dtype-dependent threshold for switching slerp to linear interpolation.
    Uses 1 - sqrt(eps): ~0.969 for f16, ~0.9997 for f32, ~1-1.5e-8 for f64."""
    return 1.0 - math.sqrt(safe_eps(dtype, xp, scale=1.0))


def random_uniform(shape: tuple[int, ...], *, dtype=None, key=None, xp: ModuleType | None = None):
    """Return U[0,1) samples of the given shape.

    Backend dispatch:
      - JAX: uses jax.random.uniform (key is required)
      - PyTorch: uses torch.rand
      - NumPy (default): uses numpy.random.uniform
    """
    if xp is None:
        import numpy as np

        arr = np.random.uniform(0, 1, shape)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    name = xp.__name__
    if "jax" in name:
        import jax.random

        if key is None:
            raise ValueError("A PRNG key is required for JAX: pass key=jax.random.PRNGKey(...)")
        kwargs = {"dtype": dtype} if dtype is not None else {}
        return jax.random.uniform(key, shape, **kwargs)
    elif "torch" in name:
        import torch

        kwargs = {"dtype": dtype} if dtype is not None else {}
        return torch.rand(shape, **kwargs)
    else:
        import numpy as np

        arr = np.random.uniform(0, 1, shape)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr


def eye(n: int, *, dtype, xp: ModuleType, like=None):
    """Identity matrix with backend-aware device placement.

    For torch, passing ``like`` keeps the identity on the same device as ``like``.
    Other backends ignore ``like`` and use the standard ``xp.eye`` behavior.
    """
    name = xp.__name__
    if "torch" in name and like is not None:
        return xp.eye(n, dtype=dtype, device=like.device)
    return xp.eye(n, dtype=dtype)
