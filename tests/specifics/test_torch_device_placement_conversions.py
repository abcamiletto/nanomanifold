import pytest

from nanomanifold import SO3


def _torch_accel_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def test_so3_conversion_outputs_stay_on_torch_accelerator():
    torch = pytest.importorskip("torch")
    device = _torch_accel_device(torch)
    if device is None:
        pytest.skip("No torch accelerator available (cuda/mps)")

    axis_angle = torch.randn(128, 3, device=device, dtype=torch.float32)
    euler = torch.randn(128, 3, device=device, dtype=torch.float32)

    rotmat_from_axis = SO3.conversions.from_axis_angle_to_rotmat(axis_angle, xp=torch)
    rotmat_from_euler = SO3.conversions.from_euler_to_rotmat(euler, xp=torch)

    assert rotmat_from_axis.device.type == device.type
    assert rotmat_from_euler.device.type == device.type
