import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PRECISIONS, random_se3

from nanomanifold import SE3, SO3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_left_jacobian_identity(backend, batch_dims, precision):
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend)

    dtype = getattr(xp, f"float{precision}")

    tangent = xp.zeros(batch_dims + (6,), dtype=dtype)
    J = SE3.left_jacobian(tangent)

    assert J.shape == batch_dims + (6, 6)

    identity = np.eye(6, dtype=np.float64)
    identity = np.broadcast_to(identity, J.shape)
    J_np = np.asarray(J, dtype=np.float64)
    assert np.allclose(J_np, identity, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_left_jacobian_inverse_pair(backend, batch_dims, precision):
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend)

    shape = batch_dims + (6,)
    tangent_np = 0.2 * np.random.normal(size=shape).astype(f"float{precision}")
    tangent = xp.asarray(tangent_np)

    J = SE3.left_jacobian(tangent)
    J_inv = SE3.left_jacobian_inverse(tangent)

    product = xp.matmul(J_inv, J)

    assert product.shape == batch_dims + (6, 6)

    identity = np.eye(6, dtype=np.float64)
    identity = np.broadcast_to(identity, product.shape)
    product_np = np.asarray(product, dtype=np.float64)

    tolerance = ATOL[precision]
    if precision == 16:
        tolerance = max(tolerance, 4e-3)

    assert np.allclose(product_np, identity, atol=tolerance)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_adjoint_matches_definition(backend, batch_dims, precision):
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend)

    se3 = random_se3(batch_dims=batch_dims, backend=backend, precision=precision)

    adjoint_matrix = SE3.adjoint(se3)

    rotation = SO3.to_matrix(se3[..., :4])
    translation = se3[..., 4:7]
    translation_hat = SO3.hat(translation)
    expected_bottom = xp.matmul(translation_hat, rotation)

    zeros = xp.zeros_like(rotation)
    expected_top = xp.concatenate([rotation, zeros], axis=-1)
    expected_bottom_row = xp.concatenate([expected_bottom, rotation], axis=-1)
    expected = xp.concatenate([expected_top, expected_bottom_row], axis=-2)

    assert adjoint_matrix.shape == expected.shape == batch_dims + (6, 6)

    adj_np = np.asarray(adjoint_matrix, dtype=np.float64)
    expected_np = np.asarray(expected, dtype=np.float64)

    tolerance = ATOL[precision]
    if precision == 16:
        tolerance = max(tolerance, 2e-2)

    assert np.allclose(adj_np, expected_np, atol=tolerance)
