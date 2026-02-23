import importlib.metadata

import nanomanifold


def test_package_exposes_version():
    assert hasattr(nanomanifold, "__version__")
    assert nanomanifold.__version__ == importlib.metadata.version("nanomanifold")
