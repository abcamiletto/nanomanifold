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
