import importlib.metadata

from . import SE3, SO3

__version__ = importlib.metadata.version("nanomanifold")

__all__ = ["SO3", "SE3", "__version__"]
