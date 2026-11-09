import numpy as np
import pytest
from conftest import ATOL, TEST_BATCH_DIMS, TEST_PRECISIONS, random_quaternion

from nanomanifold import SE3


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_rt_conversion_cycle(batch_dims, precision):
    # Create random quaternion and translation
    quat = random_quaternion(batch_dims=batch_dims, precision=precision)

    # Generate random translation

    shape = batch_dims + (3,)
    translation_np = np.random.randn(*shape).astype(f"float{precision}")
    xp = np
    translation = xp.asarray(translation_np)

    # Convert to SE(3) representation
    se3 = SE3.from_rt(quat, translation)

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


@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_rt_matrix_consistency(batch_dims):
    """Test that from_rt -> to_matrix -> from_matrix -> to_rt gives same result."""
    # Create random quaternion and translation
    quat = random_quaternion(batch_dims=batch_dims, precision=32)

    shape = batch_dims + (3,)
    translation_np = np.random.randn(*shape).astype("float32")
    xp = np
    translation = xp.asarray(translation_np)

    # Route 1: from_rt -> to_matrix -> from_matrix -> to_rt
    se3_from_rt = SE3.from_rt(quat, translation)
    matrix = SE3.to_matrix(se3_from_rt)
    se3_from_matrix = SE3.from_matrix(matrix)
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
