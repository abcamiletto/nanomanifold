import numpy as np
import pytest

from nanomanifold import common


def test_get_namespace_numpy():
    assert common.get_namespace(np.ones(1)) is np


def test_get_namespace_numpy_without_array_namespace():
    class Numpy1Array(np.ndarray):
        def __getattribute__(self, name):
            if name == "__array_namespace__":
                raise AttributeError
            return super().__getattribute__(name)

    arr = np.ones(1).view(Numpy1Array)

    assert common.get_namespace(arr) is np


def test_get_namespace_torch():
    torch = pytest.importorskip("torch")

    assert common.get_namespace(torch.ones(1)) is torch


def test_get_namespace_jax():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    assert common.get_namespace(jnp.ones(1)) is jnp


@pytest.mark.parametrize(
    ("name", "module_name"),
    [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("jax", "jax.numpy"),
    ],
)
def test_get_namespace_by_name(name, module_name):
    pytest.importorskip(module_name)

    xp = common.get_namespace_by_name(name)

    assert xp.__name__ == module_name
