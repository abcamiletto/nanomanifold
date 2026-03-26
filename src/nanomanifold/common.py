from types import ModuleType
from typing import Any


def get_namespace(array: Any) -> ModuleType:
    namespace = getattr(array, "__array_namespace__", None)
    if namespace is not None:
        return namespace()

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        if isinstance(array, np.ndarray):
            return np

    if type(array).__module__.startswith("torch"):
        import torch

        return torch

    raise TypeError(f"Unsupported array type '{type(array).__name__}'.")


def get_namespace_by_name(name: str) -> ModuleType:
    if name == "numpy":
        import numpy as np

        return np
    if name == "torch":
        import torch

        return torch
    if name == "jax":
        import jax.numpy as jnp

        return jnp
    raise ValueError(f"Unknown array namespace '{name}'. Supported: 'numpy', 'torch', 'jax'.")


def safe_eps(dtype: Any, xp: ModuleType, scale: float = 10.0) -> float:
    """Machine epsilon * scale. Used for division-by-zero guards."""
    return float(xp.finfo(dtype).eps) * scale


def zeros_as(ref: Any, *, shape: tuple[int, ...], xp: ModuleType | None = None) -> Any:
    """Create zeros matching ref backend/device/dtype with explicit shape."""
    if xp is None:
        xp = get_namespace(ref)
    if "torch" in xp.__name__:
        return xp.zeros(shape, dtype=ref.dtype, device=ref.device)
    return xp.zeros(shape, dtype=ref.dtype)


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
