import pytest

from nanomanifold import SE3, SO3


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

    rotmat_from_axis = SO3.convert(axis_angle, src="axis_angle", dst="rotmat", xp=torch)
    rotmat_from_euler = SO3.convert(euler, src="euler", dst="rotmat", xp=torch)

    assert rotmat_from_axis.device.type == device.type
    assert rotmat_from_euler.device.type == device.type


@pytest.mark.parametrize("rep", ["axis_angle", "euler", "hinge", "matrix", "rotmat", "quat", "sixd"])
def test_identity_as_stays_on_torch_accelerator(rep):
    torch = pytest.importorskip("torch")
    device = _torch_accel_device(torch)
    if device is None:
        pytest.skip("No torch accelerator available (cuda/mps)")
    ref = torch.zeros(2, 4, device=device, dtype=torch.float16)

    identity = SO3.identity_as(ref, batch_dims=(2,), rep=rep)
    transform = SE3.identity_as(ref, batch_dims=(2,))

    assert identity.device.type == device.type and identity.dtype == torch.float16
    assert transform.device.type == device.type and transform.dtype == torch.float16
