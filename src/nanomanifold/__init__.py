from importlib.metadata import PackageNotFoundError, version

from . import SE3, SO3

try:
    __version__ = version("nanomanifold")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["SO3", "SE3", "__version__"]
