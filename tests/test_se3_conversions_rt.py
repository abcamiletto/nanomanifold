import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PASS_XP, TEST_PRECISIONS, get_xp_kwargs, random_quaternion

from nanomanifold import SE3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_rt_conversion_cycle(backend, batch_dims, precision, pass_xp):
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    # Create random quaternion and translation
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    # Generate random translation
    from nanomanifold.common import get_namespace_by_name

    shape = batch_dims + (3,)
    translation_np = np.random.randn(*shape).astype(f"float{precision}")
    xp = get_namespace_by_name(backend)
    translation = xp.asarray(translation_np)

    # Convert to SE(3) representation
    se3 = SE3.from_rt(quat, translation, **xp_kwargs)

    assert se3.dtype == quat.dtype
    assert se3.shape[:-1] == quat.shape[:-1]
    assert se3.shape[-1] == 7

    # Convert back to rotation and translation
    quat_converted, translation_converted = SE3.to_rt(se3)

    assert quat_converted.dtype == quat.dtype
    assert quat_converted.shape == quat.shape
    assert translation_converted.dtype == translation.dtype
    assert translation_converted.shape == translation.shape

    # Convert to numpy arrays and compare
    quat_np = np.array(quat)
    quat_converted_np = np.array(quat_converted)
    translation_np_orig = np.array(translation)
    translation_converted_np = np.array(translation_converted)

    # Check quaternion equivalence (q and -q represent the same rotation)
    dot_products = np.sum(quat_np * quat_converted_np, axis=-1)
    assert np.allclose(np.abs(dot_products), 1.0, atol=ATOL[precision])

    # Check translation equivalence
    assert np.allclose(translation_np_orig, translation_converted_np, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_rt_matrix_consistency(backend, batch_dims, pass_xp):
    """Test that from_rt -> to_matrix -> from_matrix -> to_rt gives same result."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    # Create random quaternion and translation
    quat = random_quaternion(batch_dims=batch_dims, backend=backend, precision=32)

    from nanomanifold.common import get_namespace_by_name

    shape = batch_dims + (3,)
    translation_np = np.random.randn(*shape).astype("float32")
    xp = get_namespace_by_name(backend)
    translation = xp.asarray(translation_np)

    # Route 1: from_rt -> to_matrix -> from_matrix -> to_rt
    se3_from_rt = SE3.from_rt(quat, translation, **xp_kwargs)
    matrix = SE3.to_matrix(se3_from_rt, **xp_kwargs)
    se3_from_matrix = SE3.from_matrix(matrix, **xp_kwargs)
    quat_final, translation_final = SE3.to_rt(se3_from_matrix)

    # Convert to numpy for comparison
    quat_np = np.array(quat)
    quat_final_np = np.array(quat_final)
    translation_np = np.array(translation)
    translation_final_np = np.array(translation_final)

    # Check quaternion equivalence (q and -q represent the same rotation)
    dot_products = np.sum(quat_np * quat_final_np, axis=-1)
    assert np.allclose(np.abs(dot_products), 1.0, atol=1e-6)

    # Check translation equivalence
    assert np.allclose(translation_np, translation_final_np, atol=1e-6)


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_rt_differentiality_torch(batch_dims):
    torch = pytest.importorskip("torch")
    # Use double precision for gradient checking as recommended by PyTorch
    dtype = torch.float64

    # Random quaternion and translation inputs
    quat = random_quaternion(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)
    translation = torch.randn(batch_dims + (3,), dtype=dtype, requires_grad=True)

    # Check gradients of SE3.from_rt
    def f(q, t):
        return SE3.from_rt(q, t)

    assert torch.autograd.gradcheck(f, (quat, translation), eps=1e-6, atol=1e-5)

    # SE3 input from random quaternion and translation
    se3 = SE3.from_rt(quat.detach(), translation.detach()).requires_grad_(True)

    # Check gradients of SE3.to_rt
    def g(s):
        q, t = SE3.to_rt(s)
        return q.sum() + t.sum()  # Need scalar output for gradcheck

    assert torch.autograd.gradcheck(g, (se3,), eps=1e-6, atol=1e-5)
