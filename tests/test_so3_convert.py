import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PASS_XP, TEST_PRECISIONS, get_xp_kwargs, random_quaternion

from nanomanifold import SO3
from nanomanifold.common import get_namespace_by_name

_SOURCE_REPRESENTATIONS = ["axis_angle", "euler", "hinge", "matrix", "rotmat", "quat", "sixd"]
_TARGET_REPRESENTATIONS = ["axis_angle", "euler", "hinge", "matrix", "rotmat", "quat", "sixd"]
_PAIRS = [(src, dst) for src in _SOURCE_REPRESENTATIONS for dst in _TARGET_REPRESENTATIONS]


def _make_axes(backend, precision):
    xp = get_namespace_by_name(backend)
    return xp.asarray(np.array([0.0, 0.0, 2.0], dtype=f"float{precision}"))


def _make_input(rep: str, batch_dims, backend, precision=32, convention="ZYX", quat_convention="wxyz"):
    q = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)
    if rep == "axis_angle":
        return SO3.to_axis_angle(q)
    if rep == "euler":
        return SO3.to_euler(q, convention=convention)
    if rep == "hinge":
        return SO3.to_hinge(q, _make_axes(backend, precision))
    if rep == "matrix":
        return SO3.to_rotmat(q)
    if rep == "rotmat":
        return SO3.to_rotmat(q)
    if rep == "quat":
        return SO3.to_quat(q, convention=quat_convention)
    if rep == "sixd":
        return SO3.to_sixd(q)
    raise ValueError(rep)


def _manual_convert(x, source, target, axes):
    if source == target:
        return x

    if source == "axis_angle":
        q = SO3.from_axis_angle(x)
    elif source == "euler":
        q = SO3.from_euler(x)
    elif source == "hinge":
        q = SO3.from_hinge(x, axes)
    elif source == "matrix":
        q = SO3.from_matrix(x)
    elif source == "rotmat":
        q = SO3.from_rotmat(x)
    elif source == "quat":
        q = SO3.from_quat(x)
    else:
        q = SO3.from_sixd(x)

    if target == "axis_angle":
        return SO3.to_axis_angle(q)
    if target == "euler":
        return SO3.to_euler(q)
    if target == "hinge":
        return SO3.to_hinge(q, axes)
    if target == "matrix":
        return SO3.to_rotmat(q)
    if target == "rotmat":
        return SO3.to_rotmat(q)
    if target == "quat":
        return SO3.to_quat(q)
    return SO3.to_sixd(q)


@pytest.mark.parametrize("source,target", _PAIRS, ids=[f"{s}->{t}" for s, t in _PAIRS])
@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_convert_matches_manual_two_stage_dispatch(source, target, backend, batch_dims, precision, pass_xp):
    axes = _make_axes(backend, precision)
    x = _make_input(
        source,
        batch_dims,
        backend,
        precision=precision,
    )
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    convert_kwargs = {}
    if source == "hinge" and target != "hinge":
        convert_kwargs["src_kwargs"] = {"axes": axes}
    if target == "hinge" and source != "hinge":
        convert_kwargs["dst_kwargs"] = {"axes": axes}

    if precision == 16 and source == "matrix":
        with pytest.raises(ValueError, match="does not support float16"):
            SO3.convert(
                x,
                src=source,
                dst=target,
                **convert_kwargs,
                **xp_kwargs,
            )
        return

    result = SO3.convert(
        x,
        src=source,
        dst=target,
        **convert_kwargs,
        **xp_kwargs,
    )
    expected = _manual_convert(
        x,
        source,
        target,
        axes,
    )

    assert result.dtype == expected.dtype
    assert result.shape == expected.shape

    result_np = np.array(result)
    expected_np = np.array(expected)

    atol = ATOL[precision]
    if target == "quat":
        dots = np.sum(result_np * expected_np, axis=-1)
        assert np.allclose(np.abs(dots), 1.0, atol=atol)
    elif target == "euler":
        result_quat = SO3.from_euler(result, convention="ZYX")
        expected_quat = SO3.from_euler(expected, convention="ZYX")
        dots = np.sum(np.array(result_quat) * np.array(expected_quat), axis=-1)
        assert np.allclose(np.abs(dots), 1.0, atol=atol)
    else:
        assert np.allclose(result_np, expected_np, atol=atol)


@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_convert_supports_euler_to_euler_convention_changes(precision):
    q = random_quaternion(batch_dims=(8,), backend="numpy", precision=precision)
    euler_xyz = SO3.to_euler(q, convention="XYZ")

    converted = SO3.convert(
        euler_xyz,
        src="euler",
        dst="euler",
        src_kwargs={"convention": "XYZ"},
        dst_kwargs={"convention": "ZYX"},
    )
    expected = SO3.to_euler(SO3.from_euler(euler_xyz, convention="XYZ"), convention="ZYX")

    assert converted.dtype == expected.dtype
    assert converted.shape == expected.shape
    assert np.allclose(np.array(converted), np.array(expected), atol=ATOL[precision])


@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_convert_supports_explicit_quaternion_order_changes(precision):
    q = random_quaternion(batch_dims=(8,), backend="numpy", precision=precision)
    quat_xyzw = SO3.to_quat(q, convention="xyzw")

    converted = SO3.convert(
        quat_xyzw,
        src="quat",
        dst="quat",
        src_kwargs={"convention": "xyzw"},
        dst_kwargs={"convention": "wxyz"},
    )
    expected = SO3.canonicalize(q)

    assert converted.dtype == expected.dtype
    assert converted.shape == expected.shape
    assert np.allclose(np.array(converted), np.array(expected), atol=ATOL[precision])


def test_convert_distinguishes_matrix_from_rotmat():
    quat = random_quaternion(batch_dims=(4,), backend="numpy", precision=32)
    rotmat = SO3.to_rotmat(quat)
    stretch = np.diag(np.array([1.05, 0.97, 1.02], dtype=np.float32))
    matrix = np.matmul(np.array(rotmat), stretch)

    converted = SO3.convert(matrix, src="matrix", dst="rotmat")
    expected = SO3.conversions.from_matrix_to_rotmat(matrix)

    assert np.allclose(np.array(converted), np.array(expected), atol=ATOL[32])


def test_convert_projects_matrix_before_quaternion_conversion():
    quat = random_quaternion(batch_dims=(4,), backend="numpy", precision=32)
    rotmat = SO3.to_rotmat(quat)
    stretch = np.diag(np.array([1.05, 0.97, 1.02], dtype=np.float32))
    matrix = np.matmul(np.array(rotmat), stretch)

    converted = SO3.convert(matrix, src="matrix", dst="quat")
    expected = SO3.conversions.from_rotmat_to_quat(SO3.conversions.from_matrix_to_rotmat(matrix))

    dots = np.sum(np.array(converted) * np.array(expected), axis=-1)
    assert np.allclose(np.abs(dots), 1.0, atol=ATOL[32])


def test_convert_matrix_output_matches_rotmat():
    quat = random_quaternion(batch_dims=(4,), backend="numpy", precision=32)

    matrix = SO3.convert(quat, src="quat", dst="matrix")
    rotmat = SO3.convert(quat, src="quat", dst="rotmat")

    assert np.allclose(np.array(matrix), np.array(rotmat), atol=ATOL[32])


def test_convert_rejects_float16_matrix_projection():
    quat = random_quaternion(batch_dims=(4,), backend="numpy", precision=16)
    matrix = SO3.to_rotmat(quat)

    with pytest.raises(ValueError, match="does not support float16"):
        SO3.convert(matrix, src="matrix", dst="rotmat")


def test_convert_requires_hinge_axes():
    with pytest.raises(TypeError):
        SO3.convert(np.array([[0.1]], dtype=np.float32), src="hinge", dst="quat")
