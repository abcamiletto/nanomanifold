import numpy as np
import pytest
from conftest import (
    ATOL,
    TEST_BACKENDS,
    TEST_BATCH_DIMS,
    TEST_PRECISIONS,
    identity_quaternion,
)

from nanomanifold import SO3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_left_jacobian_identity(backend, batch_dims, precision):
    from nanomanifold.common import get_namespace_by_name

    xp = get_namespace_by_name(backend)

    dtype = getattr(xp, f"float{precision}")

    omega = xp.zeros(batch_dims + (3,), dtype=dtype)
    J = SO3.left_jacobian(omega)

    assert J.shape == batch_dims + (3, 3)

    identity = np.eye(3, dtype=np.float64)
    identity = np.broadcast_to(identity, J.shape)
    J_np = np.asarray(J, dtype=np.float64)
    assert np.allclose(J_np, identity, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_left_jacobian_inverse_pair(backend, batch_dims, precision):
    from nanomanifold.common import get_namespace_by_name

    xp = get_namespace_by_name(backend)

    shape = batch_dims + (3,)
    omega_np = 0.2 * np.random.normal(size=shape).astype(f"float{precision}")
    omega = xp.asarray(omega_np)

    J = SO3.left_jacobian(omega)
    J_inv = SO3.left_jacobian_inverse(omega)

    product = xp.matmul(J_inv, J)

    assert product.shape == batch_dims + (3, 3)

    identity = np.eye(3, dtype=np.float64)
    identity = np.broadcast_to(identity, product.shape)
    product_np = np.asarray(product, dtype=np.float64)

    assert np.allclose(product_np, identity, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_adjoint_matches_matrix(backend, batch_dims, precision):
    q = identity_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)

    adjoint_matrix = SO3.adjoint(q)
    rotation_matrix = SO3.to_matrix(q)

    assert adjoint_matrix.shape == rotation_matrix.shape == batch_dims + (3, 3)

    adj_np = np.asarray(adjoint_matrix, dtype=np.float64)
    rot_np = np.asarray(rotation_matrix, dtype=np.float64)
    assert np.allclose(adj_np, rot_np, atol=ATOL[precision])
