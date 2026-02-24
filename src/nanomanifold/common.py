from types import ModuleType
from typing import Any

import array_api_compat


def get_namespace(array: Any) -> ModuleType:
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


def safe_eps(dtype: Any, xp: ModuleType, scale: float = 10.0) -> float:
    """Machine epsilon * scale. Used for division-by-zero guards."""
    return float(xp.finfo(dtype).eps) * scale


def zeros_as(ref: Any, *, shape: tuple[int, ...]) -> Any:
    """Create zeros matching ref backend/device/dtype with explicit shape."""
    xp = get_namespace(ref)
    if "torch" in xp.__name__:
        return xp.zeros(shape, dtype=ref.dtype, device=ref.device)
    return xp.zeros(shape, dtype=ref.dtype)


def eye_as(ref: Any, *, batch_dims: tuple[int, ...]) -> Any:
    """Create batched identity matrices matching ref backend/device/dtype."""
    xp = get_namespace(ref)
    n = ref.shape[-1]
    if "torch" in xp.__name__:
        eye = xp.eye(n, dtype=ref.dtype, device=ref.device)
    else:
        eye = xp.eye(n, dtype=ref.dtype)
    return xp.broadcast_to(eye, (*batch_dims, n, n))


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
