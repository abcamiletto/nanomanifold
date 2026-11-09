"""Group operations at the configurations that break naive code (see ``conftest.awkward_quaternions``), on NumPy."""

import numpy as np
import pytest
from conftest import TEST_PRECISIONS, awkward_quaternions
from scipy.spatial.transform import Rotation as R

from nanomanifold import SE3, SO3

TOL = {16: 3e-2, 32: 1e-5}
BEYOND_PI = np.array([np.pi + 1e-3, 1.5 * np.pi, 2 * np.pi - 1e-3, 2 * np.pi + 1e-3, 3 * np.pi])


def _same_rotation(q1, q2, tol):
    return np.allclose(np.abs(np.sum(q1 * q2, axis=-1)), 1.0, atol=tol)


def _se3(precision):
    """Awkward rotations with zero, unit-scale and large translations."""
    quats = awkward_quaternions()
    translations = np.random.default_rng(0).normal(size=(len(quats), 3))
    se3 = np.concatenate([np.concatenate([quats, scale * translations], axis=-1) for scale in (0.0, 1.0, 100.0)])
    return se3.astype(f"float{precision}")


def _beyond_pi_tangents():
    axes = SO3.to_axis_angle(awkward_quaternions()[-8:])
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    return (BEYOND_PI[:, None, None] * axes[None]).reshape(-1, 3)


def _same_transform(T1, T2, tol):
    """Compare as matrices, with a tolerance relative to each transform's largest entry."""
    M1 = SE3.to_matrix(T1.astype(np.float64))
    M2 = SE3.to_matrix(T2.astype(np.float64))
    scale = 1 + np.abs(M2).max(axis=(-2, -1), keepdims=True)
    return np.all(np.abs(M1 - M2) <= tol * scale)


@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_so3_log_is_minimal_and_inverts_exp(precision):
    quat = awkward_quaternions(precision)

    for same_rotation in (quat, -quat):
        tangent = SO3.log(same_rotation)

        assert tangent.dtype == quat.dtype
        assert np.all(np.linalg.norm(tangent.astype(np.float64), axis=-1) <= np.pi + TOL[precision])
        assert _same_rotation(SO3.exp(tangent), quat, TOL[precision])


@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_so3_exp_wraps_tangents_beyond_pi(precision):
    tangent = _beyond_pi_tangents().astype(f"float{precision}")

    quat = SO3.exp(tangent)
    wrapped = SO3.log(quat)

    expected = R.from_rotvec(tangent.astype(np.float64)).as_quat()
    assert _same_rotation(quat, np.concatenate([expected[:, 3:], expected[:, :3]], axis=-1), TOL[precision])
    assert np.all(np.linalg.norm(wrapped.astype(np.float64), axis=-1) <= np.pi + TOL[precision])
    assert _same_rotation(SO3.exp(wrapped), quat, TOL[precision])


@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_so3_inverse_undoes_rotation(precision):
    quat = awkward_quaternions(precision)
    points = np.random.default_rng(0).normal(size=(len(quat), 5, 3)).astype(quat.dtype)
    identity = np.broadcast_to(np.array([1, 0, 0, 0], dtype=quat.dtype), quat.shape)

    inverse = SO3.inverse(quat)

    assert _same_rotation(SO3.multiply(quat, inverse), identity, TOL[precision])
    assert np.allclose(SO3.rotate_points(inverse, SO3.rotate_points(quat, points)), points, atol=TOL[precision] * 10)


@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_se3_log_inverts_exp(precision):
    se3 = _se3(precision)
    sign_flipped = np.concatenate([-se3[..., :4], se3[..., 4:]], axis=-1)

    for same_transform in (se3, sign_flipped):
        tangent = SE3.log(same_transform)

        assert tangent.dtype == se3.dtype
        assert np.all(np.linalg.norm(tangent[..., :3].astype(np.float64), axis=-1) <= np.pi + TOL[precision])
        assert _same_transform(SE3.exp(tangent), se3, TOL[precision])


@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_se3_exp_wraps_tangents_beyond_pi(precision):
    omega = _beyond_pi_tangents()
    rho = np.random.default_rng(0).normal(size=omega.shape)
    tangent = np.concatenate([omega, rho], axis=-1).astype(f"float{precision}")

    se3 = SE3.exp(tangent)
    wrapped = SE3.log(se3)

    assert np.all(np.linalg.norm(wrapped[..., :3].astype(np.float64), axis=-1) <= np.pi + TOL[precision])
    assert _same_transform(SE3.exp(wrapped), se3, TOL[precision])


@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_se3_inverse_undoes_transform(precision):
    se3 = _se3(precision)
    points = np.random.default_rng(1).normal(size=(len(se3), 5, 3)).astype(se3.dtype)
    identity = np.zeros_like(se3)
    identity[..., 0] = 1

    inverse = SE3.inverse(se3)

    assert _same_transform(SE3.multiply(se3, inverse), identity, TOL[precision] * 100)
    restored = SE3.transform_points(inverse, SE3.transform_points(se3, points))
    assert np.allclose(restored, points, rtol=TOL[precision], atol=TOL[precision] * 100)
