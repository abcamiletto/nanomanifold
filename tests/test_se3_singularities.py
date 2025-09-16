import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PRECISIONS, identity_se3, random_se3

from nanomanifold import SE3, SO3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_exact_pi_rotation_with_translation(backend, precision):
    """Test SE3 logarithm with exactly π rotation and various translations."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend.replace("jax", "jax"))
    
    # π rotations around principal axes with different translations
    test_cases = [
        # (quaternion, translation)
        ([0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),  # π around x, translate along x
        ([0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),  # π around x, translate along y
        ([0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),  # π around y, translate along z
        ([0.0, 0.0, 0.0, 1.0], [1.0, 2.0, 3.0]),  # π around z, arbitrary translation
    ]
    
    for quat, trans in test_cases:
        # Create SE3 element
        se3 = xp.asarray(quat + trans)
        
        # Test log
        log_result = SE3.log(se3)
        log_np = np.array(log_result)
        
        # Angular part magnitude should be π
        omega = log_np[:3]
        omega_magnitude = np.linalg.norm(omega)
        assert np.allclose(omega_magnitude, np.pi, atol=ATOL[precision])
        
        # Test exp(log(se3)) = se3
        exp_result = SE3.exp(log_result)
        exp_np = np.array(exp_result)
        se3_np = np.array(se3)
        
        # Check quaternion part (up to sign)
        q_orig = se3_np[:4]
        q_result = exp_np[:4]
        dot_product = np.sum(q_orig * q_result)
        assert np.allclose(np.abs(dot_product), 1.0, atol=ATOL[precision])
        
        # Check translation part
        t_orig = se3_np[4:]
        t_result = exp_np[4:]
        assert np.allclose(t_orig, t_result, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_v_matrix_singularity(backend, precision):
    """Test the V matrix computation near singularity (rotation angle near π)."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend.replace("jax", "jax"))
    
    # Test angles approaching π
    angles_near_pi = [np.pi - 1e-1, np.pi - 1e-3, np.pi - 1e-6, np.pi]
    
    for angle in angles_near_pi:
        # Create rotation around z-axis using from_axis_angle
        axis_angle = xp.asarray([0.0, 0.0, angle])
        q = SO3.from_axis_angle(axis_angle)
        
        # Add translation
        translation = [1.0, 2.0, 3.0]
        se3 = xp.concatenate([q, xp.asarray(translation)])
        
        # Test log - should handle V^(-1) computation at singularity
        log_result = SE3.log(se3)
        log_np = np.array(log_result)
        
        # Test that result is finite
        assert np.all(np.isfinite(log_np))
        
        # Test round-trip
        exp_result = SE3.exp(log_result)
        exp_np = np.array(exp_result)
        se3_np = np.array(se3)
        
        # Tolerance degrades near singularity
        tol = max(ATOL[precision], (np.pi - angle) * 10) if angle < np.pi else ATOL[precision] * 10
        
        # Check quaternion (up to sign)
        q_dot = np.sum(se3_np[:4] * exp_np[:4])
        assert np.allclose(np.abs(q_dot), 1.0, atol=tol)
        
        # Check translation
        assert np.allclose(se3_np[4:], exp_np[4:], atol=tol)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_large_translation_with_pi_rotation(backend, precision):
    """Test SE3 with large translations and π rotations."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend.replace("jax", "jax"))
    
    # π rotation around x-axis
    q_pi = xp.asarray([0.0, 1.0, 0.0, 0.0])
    
    # Test with increasingly large translations
    translation_scales = [1.0, 10.0, 100.0, 1000.0]
    
    for scale in translation_scales:
        translation = xp.asarray([scale, -2*scale, 3*scale])
        se3 = xp.concatenate([q_pi, translation])
        
        # Test log
        log_result = SE3.log(se3)
        log_np = np.array(log_result)
        
        # Check that result is finite
        assert np.all(np.isfinite(log_np))
        
        # Angular part should still be π rotation
        omega = log_np[:3]
        omega_magnitude = np.linalg.norm(omega)
        assert np.allclose(omega_magnitude, np.pi, atol=ATOL[precision])
        
        # Test exp(log(se3)) = se3
        exp_result = SE3.exp(log_result)
        exp_np = np.array(exp_result)
        se3_np = np.array(se3)
        
        # Relative tolerance for large values
        rtol = 1e-5 if precision == 32 else 1e-10
        
        # Check quaternion (up to sign)
        q_dot = np.sum(se3_np[:4] * exp_np[:4])
        assert np.allclose(np.abs(q_dot), 1.0, atol=ATOL[precision])
        
        # Check translation with relative tolerance
        assert np.allclose(se3_np[4:], exp_np[4:], rtol=rtol, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_pure_translation_edge_cases(backend, precision):
    """Test SE3 operations with pure translations (identity rotation)."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend.replace("jax", "jax"))
    
    # Identity rotation
    q_identity = xp.asarray([1.0, 0.0, 0.0, 0.0])
    
    # Test various translation magnitudes
    translation_cases = [
        [1e-15, 1e-15, 1e-15],  # Very small
        [1e-8, 0.0, 0.0],       # Single axis, small
        [1000.0, 2000.0, 3000.0],  # Large
        [1e8, -1e8, 0.0],       # Very large
    ]
    
    for trans in translation_cases:
        se3 = xp.concatenate([q_identity, xp.asarray(trans)])
        
        # Test log
        log_result = SE3.log(se3)
        log_np = np.array(log_result)
        
        # Angular part should be zero
        omega = log_np[:3]
        assert np.allclose(omega, 0.0, atol=ATOL[precision])
        
        # Linear part should equal translation
        rho = log_np[3:]
        assert np.allclose(rho, trans, atol=ATOL[precision])
        
        # Test round-trip
        exp_result = SE3.exp(log_result)
        exp_np = np.array(exp_result)
        se3_np = np.array(se3)
        
        assert np.allclose(se3_np, exp_np, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_screw_motion_singularities(backend, precision):
    """Test screw motions (coupled rotation and translation along axis)."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend.replace("jax", "jax"))
    
    # Test screw motions with various pitches
    axes = [
        [1.0, 0.0, 0.0],  # x-axis
        [0.0, 1.0, 0.0],  # y-axis
        [0.0, 0.0, 1.0],  # z-axis
        [1.0, 1.0, 1.0],  # diagonal
    ]
    
    # Test both small and large rotations
    angles = [0.1, np.pi/2, np.pi - 0.1, np.pi]
    pitches = [0.1, 1.0, 10.0]  # Translation per radian
    
    for axis in axes:
        axis_np = np.array(axis)
        axis_np = axis_np / np.linalg.norm(axis_np)
        
        for angle in angles:
            for pitch in pitches:
                # Create screw motion tangent vector
                omega = angle * axis_np
                rho = pitch * omega  # Translation along rotation axis
                tangent = xp.asarray(np.concatenate([omega, rho]))
                
                # Test exp
                se3 = SE3.exp(tangent)
                se3_np = np.array(se3)
                
                # Verify it's a valid SE3 element
                q_norm = np.linalg.norm(se3_np[:4])
                assert np.allclose(q_norm, 1.0, atol=ATOL[precision])
                
                # Test log(exp(v)) = v (within the domain)
                if angle < np.pi:
                    log_result = SE3.log(se3)
                    log_np = np.array(log_result)
                    tangent_np = np.array(tangent)
                    
                    tol = ATOL[precision] * (1 + 10 * (angle / np.pi))
                    assert np.allclose(log_np, tangent_np, atol=tol)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_se3_multiplication_at_singularities(backend, batch_dims, precision):
    """Test SE3 multiplication involving singularities."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend.replace("jax", "jax"))
    
    # Create SE3 elements with π rotations
    q_pi_x = xp.asarray([0.0, 1.0, 0.0, 0.0])  # π around x
    q_pi_y = xp.asarray([0.0, 0.0, 1.0, 0.0])  # π around y
    
    # Add shape for batch_dims
    shape = batch_dims + (4,)
    q_pi_x = xp.broadcast_to(q_pi_x, shape)
    q_pi_y = xp.broadcast_to(q_pi_y, shape)
    
    # Create SE3 elements
    t1 = xp.ones(batch_dims + (3,))
    t2 = 2 * xp.ones(batch_dims + (3,))
    
    se3_1 = xp.concatenate([q_pi_x, t1], axis=-1)
    se3_2 = xp.concatenate([q_pi_y, t2], axis=-1)
    
    # Test multiplication
    product = SE3.multiply(se3_1, se3_2)
    product_np = np.array(product)
    
    # Verify product is valid
    q_product = product_np[..., :4]
    q_norms = np.linalg.norm(q_product, axis=-1)
    assert np.allclose(q_norms, 1.0, atol=ATOL[precision])
    
    # Test associativity with a third element
    se3_3 = random_se3(batch_dims, backend, precision)
    
    # (se3_1 * se3_2) * se3_3
    prod_12_3 = SE3.multiply(product, se3_3)
    
    # se3_1 * (se3_2 * se3_3)
    prod_23 = SE3.multiply(se3_2, se3_3)
    prod_1_23 = SE3.multiply(se3_1, prod_23)
    
    # Check associativity
    prod_12_3_np = np.array(prod_12_3)
    prod_1_23_np = np.array(prod_1_23)
    
    # Quaternion part (up to sign)
    q_12_3 = prod_12_3_np[..., :4]
    q_1_23 = prod_1_23_np[..., :4]
    q_dots = np.sum(q_12_3 * q_1_23, axis=-1)
    assert np.allclose(np.abs(q_dots), 1.0, atol=ATOL[precision])
    
    # Translation part
    t_12_3 = prod_12_3_np[..., 4:]
    t_1_23 = prod_1_23_np[..., 4:]
    assert np.allclose(t_12_3, t_1_23, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_se3_inverse_at_singularities(backend, precision):
    """Test SE3 inverse operation at singularities."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend.replace("jax", "jax"))
    
    # Test cases with π rotations
    test_cases = [
        ([0.0, 1.0, 0.0, 0.0], [1.0, 2.0, 3.0]),  # π around x
        ([0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 5.0]), # π around y
        ([0.0, 0.0, 0.0, 1.0], [10.0, -20.0, 30.0]), # π around z
    ]
    
    for quat, trans in test_cases:
        se3 = xp.asarray(quat + trans)
        
        # Compute inverse
        se3_inv = SE3.inverse(se3)
        se3_inv_np = np.array(se3_inv)
        
        # Verify inverse properties
        # se3 * se3_inv = identity
        product = SE3.multiply(se3, se3_inv)
        product_np = np.array(product)
        
        # Should be identity
        expected_identity = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Check quaternion part (up to sign)
        q_prod = product_np[:4]
        q_identity = expected_identity[:4]
        dot_product = np.sum(q_prod * q_identity)
        assert np.allclose(np.abs(dot_product), 1.0, atol=ATOL[precision])
        
        # Check translation part (should be zero)
        t_prod = product_np[4:]
        assert np.allclose(t_prod, [0.0, 0.0, 0.0], atol=ATOL[precision])
        
        # Test se3_inv * se3 = identity
        product2 = SE3.multiply(se3_inv, se3)
        product2_np = np.array(product2)
        
        q_prod2 = product2_np[:4]
        dot_product2 = np.sum(q_prod2 * q_identity)
        assert np.allclose(np.abs(dot_product2), 1.0, atol=ATOL[precision])
        
        t_prod2 = product2_np[4:]
        assert np.allclose(t_prod2, [0.0, 0.0, 0.0], atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_se3_numerical_stability_chain(backend):
    """Test numerical stability of repeated operations near singularities."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend.replace("jax", "jax"))
    
    # Start with a near-π rotation
    angle = np.pi - 1e-4
    axis_angle_start = xp.asarray([angle, 0.0, 0.0])  # Rotation around x-axis
    q_start = SO3.from_axis_angle(axis_angle_start)
    t_start = xp.asarray([1.0, 2.0, 3.0])
    se3_start = xp.concatenate([q_start, t_start])
    
    # Apply log/exp multiple times
    current = se3_start
    num_iterations = 10
    
    for i in range(num_iterations):
        # Log then exp
        tangent = SE3.log(current)
        current = SE3.exp(tangent)
        
        # Verify still valid
        current_np = np.array(current)
        q_norm = np.linalg.norm(current_np[:4])
        assert np.allclose(q_norm, 1.0, atol=1e-6)
    
    # Should still be close to original
    se3_start_np = np.array(se3_start)
    current_np = np.array(current)
    
    # Quaternion part (up to sign)
    q_dot = np.sum(se3_start_np[:4] * current_np[:4])
    assert np.allclose(np.abs(q_dot), 1.0, atol=1e-4)
    
    # Translation part
    assert np.allclose(se3_start_np[4:], current_np[4:], atol=1e-4)