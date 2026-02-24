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

    matrix_from_axis = SO3.conversions.from_axis_angle_to_matrix(axis_angle, xp=torch)
    matrix_from_euler = SO3.conversions.from_euler_to_matrix(euler, xp=torch)

    assert matrix_from_axis.device.type == device.type
    assert matrix_from_euler.device.type == device.type
