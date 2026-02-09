import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PASS_XP, TEST_PRECISIONS, get_xp_kwargs, random_quaternion
from scipy.spatial.transform import Rotation as R

from nanomanifold import SO3

ALL_EULER_CONVENTIONS = [
    "xyz",
    "xzy",
    "yxz",
    "yzx",
    "zxy",
    "zyx",
    "XYZ",
    "XZY",
    "YXZ",
    "YZX",
    "ZXY",
    "ZYX",
    "xyx",
    "xzx",
    "yxy",
    "yzy",
    "zxz",
    "zyz",
    "XYX",
    "XZX",
    "YXY",
    "YZY",
    "ZXZ",
    "ZYZ",
]


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("convention", ALL_EULER_CONVENTIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_euler_conversion_cycle(backend, batch_dims, precision, convention, pass_xp):
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    # Create a random SO3 quaternion
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    # Convert to Euler angles representation
    euler = SO3.to_euler(quat, convention=convention, **xp_kwargs)

    assert euler.dtype == quat.dtype
    assert euler.shape[:-1] == quat.shape[:-1]

    # Convert back to quaternion
    quat_converted = SO3.from_euler(euler, convention=convention, **xp_kwargs)

    assert quat_converted.dtype == quat.dtype
    assert quat_converted.shape == quat.shape

    # Convert to numpy arrays and compare
    quat_np = np.array(quat)
    quat_converted_np = np.array(quat_converted)

    # Check quaternion equivalence (q and -q represent the same rotation)
    dot_products = np.sum(quat_np * quat_converted_np, axis=-1)
    assert np.allclose(np.abs(dot_products), 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("convention", ALL_EULER_CONVENTIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_euler_conversion_scipy(backend, batch_dims, convention, pass_xp):
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    # Create a random SO3 quaternion
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=32)

    # Convert to Euler angles representation using nanomanifold
    euler = SO3.to_euler(quat, convention=convention, **xp_kwargs)

    # Convert to Euler angles representation using scipy
    quat_np = np.array(quat)
    # Convert from [w, x, y, z] to scipy's [x, y, z, w] format
    quat_scipy = np.concatenate([quat_np[..., 1:4], quat_np[..., 0:1]], axis=-1)
    r = R.from_quat(quat_scipy.reshape(-1, 4))
    euler_scipy = r.as_euler(convention).reshape(euler.shape)

    assert euler.dtype == quat.dtype
    assert euler.shape == euler_scipy.shape

    euler_np = np.array(euler)
    assert np.allclose(euler_np, euler_scipy, atol=1e-5)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("convention", ["xyz", "ZYX", "xzx", "ZXZ"])
def test_euler_differentiality_torch(batch_dims, convention):
    torch = pytest.importorskip("torch")
    # Use double precision for gradient checking as recommended by PyTorch
    dtype = torch.float64

    # Random quaternion input
    quat = random_quaternion(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)

    # Check gradients of SO3.to_euler
    def f(q):
        return SO3.to_euler(q, convention=convention)

    assert torch.autograd.gradcheck(f, (quat,), eps=1e-6, atol=1e-5)

    # Euler angles input from random quaternion
    euler = SO3.to_euler(quat.detach(), convention=convention).requires_grad_(True)

    # Check gradients of SO3.from_euler
    def g(e):
        return SO3.from_euler(e, convention=convention)

    assert torch.autograd.gradcheck(g, (euler,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("convention", ["xyz", "ZYX"])
def test_euler_differentiality_torch_at_singularity(batch_dims, convention):
    torch = pytest.importorskip("torch")
    dtype = torch.float64

    singular_angle = torch.tensor(np.pi / 2.0, dtype=dtype)
    euler = torch.zeros(batch_dims + (3,), dtype=dtype)
    euler[..., 1] = singular_angle

    quat = SO3.from_euler(euler, convention=convention).requires_grad_(True)

    euler_out = SO3.to_euler(quat, convention=convention)
    euler_out.sum().backward()
    assert torch.isfinite(quat.grad).all()

    euler = euler.clone().requires_grad_(True)
    quat_out = SO3.from_euler(euler, convention=convention)
    quat_out.sum().backward()
    assert torch.isfinite(euler.grad).all()


def test_from_euler_zero_gradient_torch():
    torch = pytest.importorskip("torch")
    euler = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
    quat = SO3.from_euler(euler, convention="xyz")
    quat.sum().backward()
    assert torch.isfinite(euler.grad).all()
