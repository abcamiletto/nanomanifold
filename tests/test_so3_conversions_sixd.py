import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PASS_XP, TEST_PRECISIONS, get_xp_kwargs, random_quaternion

from nanomanifold import SO3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_sixd_conversion_cycle(backend, batch_dims, precision, pass_xp):
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    d6 = SO3.to_6d(quat, **xp_kwargs)

    assert d6.dtype == quat.dtype
    assert d6.shape[:-1] == quat.shape[:-1]
    assert d6.shape[-1] == 6

    quat_converted = SO3.from_6d(d6, **xp_kwargs)

    assert quat_converted.dtype == quat.dtype
    assert quat_converted.shape == quat.shape

    quat_np = np.array(quat)
    quat_converted_np = np.array(quat_converted)

    if precision >= 32:
        dot_products = np.sum(quat_np * quat_converted_np, axis=-1)
        assert np.allclose(np.abs(dot_products), 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_sixd_differentiality_torch(batch_dims):
    torch = pytest.importorskip("torch")
    dtype = torch.float64

    quat = random_quaternion(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)

    def f(q):
        return SO3.to_6d(q)

    assert torch.autograd.gradcheck(f, (quat,), eps=1e-6, atol=1e-5)

    d6 = SO3.to_6d(quat.detach()).requires_grad_(True)

    def g(x):
        return SO3.from_6d(x)

    assert torch.autograd.gradcheck(g, (d6,), eps=1e-6, atol=1e-5)
