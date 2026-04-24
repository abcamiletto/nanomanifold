import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_PASS_XP, TEST_PRECISIONS, get_xp_kwargs

from nanomanifold import SO3
from nanomanifold.common import get_namespace_by_name


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_from_hinge_matches_axis_angle_to_quat_conversion(backend, precision, pass_xp):
    xp = get_namespace_by_name(backend)
    angles = xp.asarray(np.array([[0.1], [-0.2], [0.3]], dtype=f"float{precision}"))
    axes = xp.asarray(np.array([0.0, 0.0, 1.0], dtype=f"float{precision}"))
    xp_kwargs = get_xp_kwargs(backend, pass_xp)

    result = SO3.from_hinge(angles, axes, **xp_kwargs)
    expected = SO3.convert(angles * axes, src="axis_angle", dst="quat", **xp_kwargs)

    assert result.dtype == expected.dtype
    assert result.shape == expected.shape
    assert np.allclose(np.array(result), np.array(expected), atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_from_hinge_broadcasts_batch_axes(backend, precision, pass_xp):
    xp = get_namespace_by_name(backend)
    angles = xp.asarray(np.array([[0.1], [-0.2], [0.3]], dtype=f"float{precision}"))
    axes = xp.asarray(np.array([[0.0, 1.0, 0.0]], dtype=f"float{precision}"))
    xp_kwargs = get_xp_kwargs(backend, pass_xp)

    result = SO3.from_hinge(angles, axes, **xp_kwargs)
    expected = SO3.convert(angles * axes, src="axis_angle", dst="quat", **xp_kwargs)

    assert result.dtype == expected.dtype
    assert result.shape == expected.shape
    assert np.allclose(np.array(result), np.array(expected), atol=ATOL[precision])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_to_hinge_roundtrips_from_hinge_for_nonunit_axes(backend, precision, pass_xp):
    xp = get_namespace_by_name(backend)
    angles = xp.asarray(np.array([[0.1], [-0.2], [0.3]], dtype=f"float{precision}"))
    axes = xp.asarray(np.array([0.0, 0.0, 2.0], dtype=f"float{precision}"))
    xp_kwargs = get_xp_kwargs(backend, pass_xp)

    rotation = SO3.from_hinge(angles, axes, **xp_kwargs)
    recovered = SO3.to_hinge(rotation, axes, **xp_kwargs)

    assert recovered.dtype == angles.dtype
    assert recovered.shape == angles.shape
    assert np.allclose(np.array(recovered), np.array(angles), atol=ATOL[precision])


def test_to_hinge_projects_axis_angle_to_requested_axis():
    axis_angle = np.array([0.3, 0.4, 0.0], dtype=np.float32)
    rotation = SO3.convert(axis_angle, src="axis_angle", dst="quat")
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    angle = SO3.to_hinge(rotation, axis)

    assert angle.shape == (1,)
    assert np.allclose(np.array(angle), np.array([0.3], dtype=np.float32), atol=ATOL[32])


def test_to_hinge_returns_trailing_singleton():
    angles = np.array([[0.1], [-0.2], [0.3]], dtype=np.float32)
    axes = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    rotation = SO3.from_hinge(angles, axes)

    recovered = SO3.to_hinge(rotation, axes)

    assert recovered.shape == (3, 1)
    assert np.allclose(recovered, angles, atol=ATOL[32])


def test_hinge_helpers_accept_quaternion_convention():
    angles = np.array([[0.1], [-0.2], [0.3]], dtype=np.float32)
    axes = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    rotation = SO3.from_hinge(angles, axes, convention="xyzw")

    recovered = SO3.to_hinge(rotation, axes, convention="xyzw")

    assert np.allclose(recovered, angles, atol=ATOL[32])
