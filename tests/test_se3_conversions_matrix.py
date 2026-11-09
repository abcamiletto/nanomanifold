import numpy as np
import pytest
from conftest import ATOL, TEST_BATCH_DIMS, TEST_PRECISIONS, random_se3

from nanomanifold import SE3


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_matrix_conversion_cycle(batch_dims, precision):
    se3 = random_se3(batch_dims=batch_dims, precision=precision)

    matrix = SE3.to_matrix(se3)

    assert matrix.dtype == se3.dtype
    assert matrix.shape[:-2] == se3.shape[:-1]
    assert matrix.shape[-2:] == (4, 4)

    se3_converted = SE3.from_matrix(matrix)

    assert se3_converted.dtype == se3.dtype
    assert se3_converted.shape == se3.shape

    se3_np = np.array(se3)
    se3_converted_np = np.array(se3_converted)

    # Check quaternion equivalence (q and -q represent the same rotation)
    quat_orig = se3_np[..., :4]
    quat_conv = se3_converted_np[..., :4]
    dot_products = np.sum(quat_orig * quat_conv, axis=-1)
    assert np.allclose(np.abs(dot_products), 1.0, atol=ATOL[precision])

    trans_orig = se3_np[..., 4:7]
    trans_conv = se3_converted_np[..., 4:7]
    assert np.allclose(trans_orig, trans_conv, atol=ATOL[precision])


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_matrix_properties(batch_dims):
    se3 = random_se3(batch_dims=batch_dims, precision=32)

    matrix = SE3.to_matrix(se3)
    matrix_np = np.array(matrix)

    expected_bottom_row = np.zeros(matrix_np.shape[:-2] + (4,))
    expected_bottom_row[..., 3] = 1
    assert np.allclose(matrix_np[..., 3, :], expected_bottom_row, atol=1e-6)

    R = matrix_np[..., :3, :3]
    RTR = np.matmul(np.transpose(R, axes=list(range(R.ndim - 2)) + [-1, -2]), R)
    I = np.eye(3)
    I_batched = np.broadcast_to(I, R.shape[:-2] + (3, 3))
    assert np.allclose(RTR, I_batched, atol=1e-6)

    det = np.linalg.det(R)
    assert np.allclose(det, 1.0, atol=1e-6)
