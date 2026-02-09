import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PASS_XP, TEST_PRECISIONS, get_xp_kwargs, random_se3

from nanomanifold import SE3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_canonicalize_w_nonnegative(backend, batch_dims, precision, pass_xp):
    """Test that canonicalize ensures w >= 0."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    result = SE3.canonicalize(se3, **xp_kwargs)

    assert result.dtype == se3.dtype
    assert result.shape == se3.shape

    result_np = np.array(result)
    assert np.all(result_np[..., 0] >= -ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_canonicalize_unit_quaternion(backend, batch_dims, precision, pass_xp):
    """Test that canonicalize preserves unit quaternion norm."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    result = SE3.canonicalize(se3, **xp_kwargs)
    result_np = np.array(result)

    quat_norms = np.linalg.norm(result_np[..., :4], axis=-1)
    assert np.allclose(quat_norms, 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_canonicalize_preserves_translation(backend, batch_dims, precision, pass_xp):
    """Test that canonicalize does not modify the translation part."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    result = SE3.canonicalize(se3, **xp_kwargs)

    se3_np = np.array(se3)
    result_np = np.array(result)

    assert np.allclose(se3_np[..., 4:], result_np[..., 4:], atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_canonicalize_negated_quaternion(backend, batch_dims, precision, pass_xp):
    """Test that canonicalize(se3) and canonicalize(-q, t) give the same result."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    common = __import__("nanomanifold.common", fromlist=["get_namespace_by_name"])
    xp = common.get_namespace_by_name(backend)

    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    # Negate only the quaternion part
    quat_neg = -se3[..., :4]
    translation = se3[..., 4:]
    se3_neg = xp.concatenate([quat_neg, translation], axis=-1)

    result1 = SE3.canonicalize(se3, **xp_kwargs)
    result2 = SE3.canonicalize(se3_neg, **xp_kwargs)

    result1_np = np.array(result1)
    result2_np = np.array(result2)

    assert np.allclose(result1_np, result2_np, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_canonicalize_idempotent(backend, batch_dims, precision, pass_xp):
    """Test that applying canonicalize twice gives the same result."""
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    result1 = SE3.canonicalize(se3, **xp_kwargs)
    result2 = SE3.canonicalize(result1, **xp_kwargs)

    result1_np = np.array(result1)
    result2_np = np.array(result2)

    assert np.allclose(result1_np, result2_np, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_canonicalize_differentiability_torch(batch_dims):
    """Test differentiability of SE3.canonicalize."""
    torch = pytest.importorskip("torch")
    dtype = torch.float64

    se3 = random_se3(batch_dims=batch_dims, backend="torch").to(dtype).requires_grad_(True)

    def f(x):
        return SE3.canonicalize(x)

    assert torch.autograd.gradcheck(f, (se3,), eps=1e-6, atol=1e-5)
