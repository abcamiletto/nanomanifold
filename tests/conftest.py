import numpy as np
import pytest

from nanomanifold import SO3

TEST_PRECISIONS = [16, 32]
TEST_BATCH_DIMS = [(), (2, 3)]

ATOL = {
    16: 2e-2,
    32: 2e-6,
    64: 1e-12,
}


@pytest.fixture(autouse=True)
def seed_numpy():
    """Every test draws the same random inputs, whatever order or worker it runs in."""
    np.random.seed(0)


def awkward_quaternions(precision=64):
    """Canonical wxyz quaternions mixing random rotations with configurations that break naive code.

    Covers the identity, tiny angles, exact and near half turns about coordinate and oblique axes, and
    ZYX gimbal lock.
    """
    axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, -2, 0.5]], dtype=np.float64)
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    angles = np.array([0.0, 1e-4, np.pi - 1e-3, np.pi])
    axis_angles = (angles[:, None, None] * axes[None]).reshape(-1, 3)
    gimbal_lock = np.array([[0.3, np.pi / 2, -0.7], [-1.2, -np.pi / 2, 0.4]])
    rng = np.random.default_rng(0)
    random = rng.normal(size=(8, 4))

    quats = np.concatenate(
        [
            SO3.from_axis_angle(axis_angles),
            SO3.from_euler(gimbal_lock, convention="ZYX"),
            SO3.canonicalize(random),
        ]
    )
    return quats.astype(f"float{precision}")


@pytest.fixture(autouse=True)
def skip_torch_1_cpu_float16(request):
    callspec = getattr(request.node, "callspec", None)
    if callspec is None:
        return
    if callspec.params.get("backend") != "torch" or callspec.params.get("precision") != 16:
        return

    torch = pytest.importorskip("torch")
    if int(torch.__version__.split(".", 1)[0]) < 2:
        pytest.skip("Torch 1.x does not support these CPU float16 tests.")


def without_singularities(quat, rep, options):
    """Drop the inputs where converting to ``rep`` is legitimately ambiguous.

    Hinge targets project an axis-angle vector whose sign is arbitrary at exact half turns.
    """
    if rep == "hinge":
        return quat[np.abs(quat[..., 0]) > 1e-6]
    return quat


def random_quaternion(batch_dims=(), precision=32):
    """Random canonical wxyz quaternions of shape ``(*batch_dims, 4)``."""
    q = np.random.randn(*batch_dims, 4)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(q[..., :1] < 0, -q, q).astype(f"float{precision}")


def identity_quaternion(batch_dims=(), precision=32):
    q = np.zeros((*batch_dims, 4), dtype=f"float{precision}")
    q[..., 0] = 1
    return q


def random_points(batch_dims=(), n_points=10, precision=32):
    return np.random.randn(*batch_dims, n_points, 3).astype(f"float{precision}")


def random_se3(batch_dims=(), precision=32):
    """Random ``[quaternion, translation]`` arrays of shape ``(*batch_dims, 7)``."""
    translation = np.random.randn(*batch_dims, 3).astype(f"float{precision}")
    return np.concatenate([random_quaternion(batch_dims, precision), translation], axis=-1)


def identity_se3(batch_dims=(), precision=32):
    return np.concatenate([identity_quaternion(batch_dims, precision), np.zeros((*batch_dims, 3), dtype=f"float{precision}")], axis=-1)
