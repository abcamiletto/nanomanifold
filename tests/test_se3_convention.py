import numpy as np

from nanomanifold import SE3, SO3


def _xyzw(se3):
    quat = SO3.to_quat(se3[..., :4], convention="xyzw")
    return np.concatenate([np.array(quat), np.array(se3[..., 4:7])], axis=-1)


def test_se3_matrix_conversions_support_xyzw():
    se3 = SE3.random(8)
    se3_xyzw = _xyzw(se3)

    quat, translation = SE3.to_rt(se3_xyzw, convention="xyzw")
    matrix = SE3.to_matrix(se3_xyzw, convention="xyzw")
    recovered = SE3.from_matrix(matrix, convention="xyzw")

    assert np.allclose(np.array(quat), se3_xyzw[..., :4], atol=1e-6)
    assert np.allclose(np.array(translation), se3_xyzw[..., 4:7], atol=1e-6)
    assert np.allclose(np.array(SE3.to_matrix(recovered, convention="xyzw")), np.array(matrix), atol=1e-6)


def test_se3_operations_support_xyzw():
    se3_1 = SE3.random(8)
    se3_2 = SE3.random(8)
    se3_1_xyzw = _xyzw(se3_1)
    se3_2_xyzw = _xyzw(se3_2)
    points = np.random.randn(8, 5, 3).astype(np.float32)

    inv_xyzw = SE3.inverse(se3_1_xyzw, convention="xyzw")
    mul_xyzw = SE3.multiply(se3_1_xyzw, se3_2_xyzw, convention="xyzw")
    points_xyzw = SE3.transform_points(se3_1_xyzw, points, convention="xyzw")

    assert np.allclose(np.array(SE3.to_matrix(inv_xyzw, convention="xyzw")), np.array(SE3.to_matrix(SE3.inverse(se3_1))), atol=1e-6)
    assert np.allclose(np.array(SE3.to_matrix(mul_xyzw, convention="xyzw")), np.array(SE3.to_matrix(SE3.multiply(se3_1, se3_2))), atol=1e-6)
    assert np.allclose(np.array(points_xyzw), np.array(SE3.transform_points(se3_1, points)), atol=1e-6)


def test_se3_log_exp_support_xyzw():
    se3 = SE3.random(8)
    se3_xyzw = _xyzw(se3)

    tangent = SE3.log(se3_xyzw, convention="xyzw")
    recovered = SE3.exp(tangent, convention="xyzw")

    assert np.allclose(np.array(SE3.to_matrix(recovered, convention="xyzw")), np.array(SE3.to_matrix(se3)), atol=1e-5)


def test_se3_interpolation_and_means_support_xyzw():
    se3_1 = SE3.random(8)
    se3_2 = SE3.random(8)
    se3_1_xyzw = _xyzw(se3_1)
    se3_2_xyzw = _xyzw(se3_2)
    times = np.array([0.0, 1.0], dtype=np.float32)
    weights = np.broadcast_to(np.array([0.25, 0.75], dtype=np.float32), (8, 2))

    interp_xyzw = SE3.slerp(se3_1_xyzw, se3_2_xyzw, times, convention="xyzw")
    weighted_xyzw = SE3.weighted_mean([se3_1_xyzw, se3_2_xyzw], weights, convention="xyzw")
    mean_xyzw = SE3.mean([se3_1_xyzw, se3_2_xyzw], convention="xyzw")

    assert np.allclose(np.array(SE3.to_matrix(interp_xyzw[..., 0, :], convention="xyzw")), np.array(SE3.to_matrix(se3_1)), atol=1e-6)
    assert np.allclose(np.array(SE3.to_matrix(interp_xyzw[..., 1, :], convention="xyzw")), np.array(SE3.to_matrix(se3_2)), atol=1e-6)
    assert np.allclose(
        np.array(SE3.to_matrix(weighted_xyzw, convention="xyzw")),
        np.array(SE3.to_matrix(SE3.weighted_mean([se3_1, se3_2], weights))),
        atol=1e-6,
    )
    assert np.allclose(
        np.array(SE3.to_matrix(mean_xyzw, convention="xyzw")),
        np.array(SE3.to_matrix(SE3.mean([se3_1, se3_2]))),
        atol=1e-6,
    )


def test_se3_random_supports_xyzw():
    se3 = SE3.random(16, convention="xyzw")
    quat = SO3.from_quat(se3[..., :4], convention="xyzw")

    assert se3.shape == (16, 7)
    assert np.allclose(np.linalg.norm(np.array(quat), axis=-1), 1.0, atol=1e-6)
