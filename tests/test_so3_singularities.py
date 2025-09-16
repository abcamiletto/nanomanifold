import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PRECISIONS, random_quaternion

from nanomanifold import SO3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_exact_pi_rotation_all_axes(backend, precision):
    """Test logarithm at exactly π rotation for all principal axes."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend.replace("jax", "jax"))

    # Exactly π rotations around principal axes
    # q = cos(θ/2) + sin(θ/2) * (xi + yj + zk)
    # For θ = π: cos(π/2) = 0, sin(π/2) = 1

    pi_rotations = [
        xp.asarray([0.0, 1.0, 0.0, 0.0]),  # π around x-axis
        xp.asarray([0.0, 0.0, 1.0, 0.0]),  # π around y-axis
        xp.asarray([0.0, 0.0, 0.0, 1.0]),  # π around z-axis
    ]

    expected_logs = [
        [np.pi, 0.0, 0.0],
        [0.0, np.pi, 0.0],
        [0.0, 0.0, np.pi],
    ]

    for q, expected in zip(pi_rotations, expected_logs):
        log_result = SO3.log(q)
        log_np = np.array(log_result)

        # The magnitude should be π
        magnitude = np.linalg.norm(log_np)
        assert np.allclose(magnitude, np.pi, atol=ATOL[precision])

        # The direction should match the expected axis
        direction = log_np / magnitude
        expected_direction = np.array(expected) / np.pi

        # Check direction (might be negated due to axis ambiguity)
        dot_product = np.dot(direction, expected_direction)
        assert np.allclose(np.abs(dot_product), 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_exact_pi_rotation_arbitrary_axes(backend, precision):
    """Test logarithm at exactly π rotation for arbitrary axes."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend.replace("jax", "jax"))

    # π rotations around arbitrary normalized axes
    axes = [
        [1.0, 1.0, 0.0],  # 45° between x and y
        [1.0, 0.0, 1.0],  # 45° between x and z
        [0.0, 1.0, 1.0],  # 45° between y and z
        [1.0, 1.0, 1.0],  # diagonal
        [2.0, -1.0, 3.0],  # arbitrary
    ]

    for axis in axes:
        # Normalize axis
        axis_np = np.array(axis)
        axis_np = axis_np / np.linalg.norm(axis_np)

        # Create quaternion for π rotation: q = [0, axis]
        q = xp.asarray([0.0, axis_np[0], axis_np[1], axis_np[2]])

        # Test log
        log_result = SO3.log(q)
        log_np = np.array(log_result)

        # Magnitude should be π
        magnitude = np.linalg.norm(log_np)
        assert np.allclose(magnitude, np.pi, atol=ATOL[precision])

        # Test that exp(log(q)) = q (up to sign)
        exp_result = SO3.exp(log_result)
        exp_np = np.array(exp_result)

        q_np = np.array(q)
        dot_product = np.sum(q_np * exp_np)
        assert np.allclose(np.abs(dot_product), 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_near_pi_rotation_stability(backend, precision):
    """Test numerical stability as rotation approaches π from different directions."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend.replace("jax", "jax"))

    # Test angles approaching π
    epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    for eps in epsilons:
        # Angles just below and above π
        angles = [np.pi - eps, np.pi + eps]

        for angle in angles:
            # Rotation around z-axis using from_axis_angle
            axis_angle = xp.asarray([0.0, 0.0, angle])
            q = SO3.from_axis_angle(axis_angle)

            # Test log
            log_result = SO3.log(q)
            log_np = np.array(log_result)

            # Test exp(log(q)) = q
            exp_result = SO3.exp(log_result)
            exp_np = np.array(exp_result)

            q_np = np.array(q)
            dot_product = np.sum(q_np * exp_np)

            # Tolerance should degrade gracefully as we approach singularity
            tol = max(ATOL[precision], eps * 10)
            assert np.allclose(np.abs(dot_product), 1.0, atol=tol)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_quaternion_double_cover_mean(backend, batch_dims, precision):
    """Test mean computation with quaternion double cover (q and -q)."""
    # Create random quaternions
    q1 = random_quaternion(batch_dims, backend, precision)
    q2 = random_quaternion(batch_dims, backend, precision)
    q3 = random_quaternion(batch_dims, backend, precision)

    # Test different sign combinations
    sign_combinations = [
        [q1, q2, q3],
        [q1, q2, -q3],
        [q1, -q2, q3],
        [q1, -q2, -q3],
        [-q1, q2, q3],
        [-q1, q2, -q3],
        [-q1, -q2, q3],
        [-q1, -q2, -q3],
    ]

    results = []
    for quats in sign_combinations:
        mean_result = SO3.mean(quats)
        results.append(np.array(mean_result))

    # All means should represent the same rotation (up to sign)
    reference = results[0]
    for i in range(1, len(results)):
        dot_product = np.sum(reference * results[i], axis=-1)
        assert np.allclose(np.abs(dot_product), 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_slerp_exact_antipodal(backend, precision):
    """Test SLERP with exactly antipodal quaternions."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend.replace("jax", "jax"))

    # Test multiple antipodal pairs
    pairs = [
        (xp.asarray([1.0, 0.0, 0.0, 0.0]), xp.asarray([-1.0, 0.0, 0.0, 0.0])),  # identity and -identity
        (xp.asarray([0.0, 1.0, 0.0, 0.0]), xp.asarray([0.0, -1.0, 0.0, 0.0])),  # π around x and -π around x
        (xp.asarray([0.5, 0.5, 0.5, 0.5]), xp.asarray([-0.5, -0.5, -0.5, -0.5])),  # normalized diagonal
    ]

    t_values = xp.asarray([0.0, 0.25, 0.5, 0.75, 1.0])

    for q1, q2 in pairs:
        result = SO3.slerp(q1, q2, t_values)
        result_np = np.array(result)

        # All results should be unit quaternions
        norms = np.linalg.norm(result_np, axis=-1)
        assert np.allclose(norms, 1.0, atol=ATOL[precision])

        # Check endpoints
        assert np.allclose(np.abs(np.sum(result_np[0] * np.array(q1))), 1.0, atol=ATOL[precision])
        assert np.allclose(np.abs(np.sum(result_np[-1] * np.array(q2))), 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_log_exp_at_branch_cut(backend, precision):
    """Test log/exp behavior at the branch cut (rotations by exactly π)."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend.replace("jax", "jax"))

    # Create rotations at different positions on the branch cut
    # These are π rotations around axes in the xy-plane
    num_angles = 8
    for i in range(num_angles):
        angle = 2 * np.pi * i / num_angles
        axis = np.array([np.cos(angle), np.sin(angle), 0.0])

        # Create π rotation quaternion: q = [0, axis]
        q = xp.asarray([0.0, axis[0], axis[1], axis[2]])

        # Log should give magnitude π
        log_result = SO3.log(q)
        log_np = np.array(log_result)
        magnitude = np.linalg.norm(log_np)
        assert np.allclose(magnitude, np.pi, atol=ATOL[precision])

        # Multiple applications of log/exp should be stable
        current = q
        for _ in range(5):
            current = SO3.exp(SO3.log(current))

        current_np = np.array(current)
        q_np = np.array(q)
        dot_product = np.sum(current_np * q_np)
        assert np.allclose(np.abs(dot_product), 1.0, atol=ATOL[precision] * 10)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_gimbal_lock_euler_conversions(backend):
    """Test Euler angle conversions at gimbal lock configurations."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend.replace("jax", "jax"))

    # Gimbal lock occurs when the middle rotation is ±90°
    # For ZYX convention, this is when pitch = ±π/2

    # Create quaternions that result in gimbal lock
    gimbal_quaternions = []

    # Pitch = π/2 (looking straight up)
    q_up = SO3.from_euler(xp.asarray([[0.0, np.pi / 2, 0.0]]), convention="ZYX")
    gimbal_quaternions.append(q_up)

    # Pitch = -π/2 (looking straight down)
    q_down = SO3.from_euler(xp.asarray([[0.0, -np.pi / 2, 0.0]]), convention="ZYX")
    gimbal_quaternions.append(q_down)

    for q in gimbal_quaternions:
        # Convert to Euler angles and back
        euler = SO3.to_euler(q, convention="ZYX")
        q_reconstructed = SO3.from_euler(euler, convention="ZYX")

        # Check that we get the same rotation (up to sign)
        q_np = np.array(q).reshape(-1, 4)
        q_reconstructed_np = np.array(q_reconstructed).reshape(-1, 4)

        for i in range(len(q_np)):
            dot_product = np.sum(q_np[i] * q_reconstructed_np[i])
            assert np.allclose(np.abs(dot_product), 1.0, atol=1e-6)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_zero_rotation_stability(backend, precision):
    """Test stability near zero rotation (identity)."""
    common = pytest.importorskip("nanomanifold.common")
    xp = common.get_namespace_by_name(backend.replace("jax", "jax"))

    # Very small rotation angles
    small_angles = [1e-8, 1e-10, 1e-12, 1e-14, 1e-15]

    for angle in small_angles:
        # Small rotation around random axis
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)

        # Create tangent vector
        tangent = xp.asarray(angle * axis)

        # Test exp
        q = SO3.exp(tangent)
        q_np = np.array(q)

        # Should be very close to identity
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        assert np.allclose(q_np, identity, atol=angle * 2)

        # Test log(exp(v)) = v
        log_result = SO3.log(q)
        log_np = np.array(log_result)
        tangent_np = np.array(tangent)

        # For very small angles, numerical precision limits accuracy
        tol = max(ATOL[precision], angle * 100)
        assert np.allclose(log_np, tangent_np, atol=tol)

