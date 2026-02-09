import numpy as np

from nanomanifold.common import get_namespace_by_name

TEST_BACKENDS = ["numpy", "torch", "jax"]
TEST_PRECISIONS = [16, 32]
TEST_BATCH_DIMS = [
    (),
    (1,),
    (5,),
    (2, 3),
    (2, 3, 4, 1, 2),
]
TEST_PASS_XP = [False, True]

ATOL = {
    16: 2e-2,
    32: 2e-6,
    64: 1e-12,
}


def get_xp_kwargs(backend, pass_xp):
    """Return xp keyword arguments based on whether to pass xp explicitly."""
    if not pass_xp:
        return {}
    return {"xp": get_namespace_by_name(backend)}


np.random.seed(0)


def random_quaternion(batch_dims=(), backend="numpy", precision=32):
    """Generate random quaternion in SO(3) for testing.

    Args:
        batch_dims: Tuple of batch dimensions
        backend: Backend to use ("numpy", "torch", "jax")
        precision: Precision (16, 32, 64)

    Returns:
        Quaternion array of shape (*batch_dims, 4) as [w, x, y, z]
    """
    # Generate in numpy first
    shape = batch_dims + (4,)
    q_np = np.random.randn(*shape).astype(f"float{precision}")

    # Normalize to get unit quaternion
    norm = np.linalg.norm(q_np, axis=-1, keepdims=True)
    q_np = q_np / norm

    # Canonicalize to w >= 0
    mask = q_np[..., 0:1] < 0
    q_np = np.where(mask, -q_np, q_np)

    xp = get_namespace_by_name(backend)
    return xp.asarray(q_np)


def identity_quaternion(batch_dims=(), backend="numpy", precision=32):
    """Generate identity quaternion for testing.

    Args:
        batch_dims: Tuple of batch dimensions
        backend: Backend to use ("numpy", "torch", "jax")
        precision: Precision (16, 32, 64)

    Returns:
        Identity quaternion array of shape (*batch_dims, 4) as [1, 0, 0, 0]
    """
    # Create identity quaternion [1, 0, 0, 0] in numpy first
    shape = batch_dims + (4,)
    q_np = np.zeros(shape, dtype=f"float{precision}")
    q_np[..., 0] = 1  # w = 1, x = y = z = 0

    xp = get_namespace_by_name(backend)
    return xp.asarray(q_np)


def random_points(batch_dims=(), n_points=10, backend="numpy", precision=32):
    """Generate random 3D points for testing.

    Args:
        batch_dims: Tuple of batch dimensions
        n_points: Number of points to generate
        backend: Backend to use ("numpy", "torch", "jax")
        precision: Precision (16, 32, 64)

    Returns:
        Points array of shape (*batch_dims, n_points, 3)
    """
    # Generate in numpy first
    shape = batch_dims + (n_points, 3)
    points_np = np.random.randn(*shape).astype(f"float{precision}")

    xp = get_namespace_by_name(backend)
    return xp.asarray(points_np)


def random_se3(batch_dims=(), backend="numpy", precision=32):
    """Generate random SE(3) transformation for testing.

    Args:
        batch_dims: Tuple of batch dimensions
        backend: Backend to use ("numpy", "torch", "jax")
        precision: Precision (16, 32, 64)

    Returns:
        SE(3) array of shape (*batch_dims, 7) as [w, x, y, z, tx, ty, tz]
    """
    # Generate random quaternion
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    # Generate random translation
    shape = batch_dims + (3,)
    translation_np = np.random.randn(*shape).astype(f"float{precision}")

    xp = get_namespace_by_name(backend)
    translation = xp.asarray(translation_np)

    # Combine quaternion and translation
    return xp.concatenate([quat, translation], axis=-1)


def identity_se3(batch_dims=(), backend="numpy", precision=32):
    """Generate identity SE(3) transformation for testing.

    Args:
        batch_dims: Tuple of batch dimensions
        backend: Backend to use ("numpy", "torch", "jax")
        precision: Precision (16, 32, 64)

    Returns:
        Identity SE(3) array of shape (*batch_dims, 7) as [1, 0, 0, 0, 0, 0, 0]
    """
    # Create identity quaternion and zero translation
    quat = identity_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    shape = batch_dims + (3,)
    translation_np = np.zeros(shape, dtype=f"float{precision}")

    xp = get_namespace_by_name(backend)
    translation = xp.asarray(translation_np)

    # Combine quaternion and translation
    return xp.concatenate([quat, translation], axis=-1)
