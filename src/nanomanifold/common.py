from types import ModuleType
from typing import Any

import numpy as np


def get_namespace(array: Any) -> ModuleType:
    namespace = getattr(array, "__array_namespace__", None)
    if namespace is not None:
        return namespace()

    if isinstance(array, np.ndarray):
        return np

    if type(array).__module__.startswith("torch"):
        import torch

        return torch

    raise TypeError(f"Unsupported array type '{type(array).__name__}'.")


def get_namespace_by_name(name: str) -> ModuleType:
    if name == "numpy":
        return np
    if name == "torch":
        import torch

        return torch
    if name == "jax":
        import jax.numpy as jnp

        return jnp
    raise ValueError(f"Unknown array namespace '{name}'. Supported: 'numpy', 'torch', 'jax'.")


def astype(x: Any, dtype: Any, xp: ModuleType) -> Any:
    """Cast ``x`` to ``dtype``, keeping it in the autograd graph (``torch.asarray`` would detach it)."""
    if "torch" in xp.__name__ and hasattr(x, "to"):
        return x.to(dtype=dtype)
    return xp.asarray(x, dtype=dtype)


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


def cross(a: Any, b: Any, xp: ModuleType) -> Any:
    """Cross product over the last axis, with broadcasting (``linalg.cross`` is missing from NumPy 1.x)."""
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return xp.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1)


def random_uniform(shape: tuple[int, ...], *, dtype=None, key=None, xp: ModuleType = np):
    """Return U[0,1) samples of the given shape: ``jax.random.uniform`` (``key`` required), ``torch.rand`` or NumPy."""
    kwargs = {} if dtype is None else {"dtype": dtype}
    if "jax" in xp.__name__:
        import jax

        if key is None:
            raise ValueError("A PRNG key is required for JAX: pass key=jax.random.PRNGKey(...)")
        return jax.random.uniform(key, shape, **kwargs)
    if "torch" in xp.__name__:
        return xp.rand(shape, **kwargs)
    samples = np.random.uniform(0, 1, shape)
    return samples if dtype is None else samples.astype(dtype)
