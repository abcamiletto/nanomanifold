import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PASS_XP, TEST_PRECISIONS, get_xp_kwargs, random_quaternion

from nanomanifold import SO3

_REPRESENTATIONS = ["axis_angle", "euler", "matrix", "rotmat", "quat", "sixd"]
_PAIRS = [(src, dst) for src in _REPRESENTATIONS for dst in _REPRESENTATIONS]


def _make_input(rep: str, batch_dims, backend, precision=32, convention="ZYX", quat_convention="wxyz"):
    q = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)
    if rep == "axis_angle":
        return SO3.to_axis_angle(q)
    if rep == "euler":
        return SO3.to_euler(q, convention=convention)
    if rep == "matrix":
        return SO3.to_rotmat(q)
    if rep == "rotmat":
        return SO3.to_rotmat(q)
    if rep == "quat":
        return SO3.to_quat_xyzw(q) if quat_convention == "xyzw" else q
    if rep == "sixd":
        return SO3.to_sixd(q)
    raise ValueError(rep)


def _manual_convert(x, source, target, src_convention="ZYX", dst_convention="ZYX"):
    if source == target:
        if source == "euler" and src_convention != dst_convention:
            return SO3.conversions.from_euler_to_euler(
                x,
                source_convention=src_convention,
                target_convention=dst_convention,
            )
        if source == "quat" and src_convention != dst_convention:
            if src_convention == "wxyz":
                return SO3.conversions.from_quat_wxyz_to_quat_xyzw(x)
            return SO3.conversions.from_quat_xyzw_to_quat_wxyz(x)
        return x

    if source == "axis_angle":
        q = SO3.from_axis_angle(x)
    elif source == "euler":
        q = SO3.from_euler(x, convention=src_convention)
    elif source == "matrix":
        q = SO3.from_matrix(x)
    elif source == "rotmat":
        q = SO3.from_rotmat(x)
    elif source == "quat":
        q = SO3.from_quat_xyzw(x) if src_convention == "xyzw" else SO3.canonicalize(x)
    else:
        q = SO3.from_sixd(x)

    if target == "axis_angle":
        return SO3.to_axis_angle(q)
    if target == "euler":
        return SO3.to_euler(q, convention=dst_convention)
    if target == "matrix":
        return SO3.to_rotmat(q)
    if target == "rotmat":
        return SO3.to_rotmat(q)
    if target == "quat":
        return SO3.to_quat_xyzw(q) if dst_convention == "xyzw" else SO3.canonicalize(q)
    return SO3.to_sixd(q)


@pytest.mark.parametrize("source,target", _PAIRS, ids=[f"{s}->{t}" for s, t in _PAIRS])
@pytest.mark.parametrize("src_quat_convention", ["wxyz", "xyzw"])
@pytest.mark.parametrize("dst_quat_convention", ["wxyz", "xyzw"])
@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_convert_matches_manual_two_stage_dispatch(
    source, target, src_quat_convention, dst_quat_convention, backend, batch_dims, precision, pass_xp
):
    src_convention = src_quat_convention if source == "quat" else "XYZ"
    dst_convention = dst_quat_convention if target == "quat" else "ZYX"
    x = _make_input(source, batch_dims, backend, precision=precision, convention="XYZ", quat_convention=src_quat_convention)
    xp_kwargs = get_xp_kwargs(backend, pass_xp)

    if precision == 16 and source == "matrix" and target != "matrix":
        with pytest.raises(ValueError, match="does not support float16"):
            SO3.convert(
                x,
                src=source,
                dst=target,
                src_convention=src_convention,
                dst_convention=dst_convention,
                **xp_kwargs,
            )
        return

    result = SO3.convert(
        x,
        src=source,
        dst=target,
        src_convention=src_convention,
        dst_convention=dst_convention,
        **xp_kwargs,
    )
    expected = _manual_convert(
        x,
        source,
        target,
        src_convention=src_convention,
        dst_convention=dst_convention,
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

    converted = SO3.convert(euler_xyz, src="euler", dst="euler", src_convention="XYZ", dst_convention="ZYX")
    expected = SO3.to_euler(SO3.from_euler(euler_xyz, convention="XYZ"), convention="ZYX")

    assert converted.dtype == expected.dtype
    assert converted.shape == expected.shape
    assert np.allclose(np.array(converted), np.array(expected), atol=ATOL[precision])


@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_convert_supports_quat_convention_changes(precision):
    q = random_quaternion(batch_dims=(8,), backend="numpy", precision=precision)
    quat_xyzw = SO3.to_quat_xyzw(q)

    converted = SO3.convert(quat_xyzw, src="quat", dst="quat", src_convention="xyzw", dst_convention="wxyz")
    expected = SO3.canonicalize(q)

    assert converted.dtype == expected.dtype
    assert converted.shape == expected.shape
    assert np.allclose(np.array(converted), np.array(expected), atol=ATOL[precision])


@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_convert_defaults_quat_to_repo_ordering(precision):
    q = random_quaternion(batch_dims=(8,), backend="numpy", precision=precision)

    converted = SO3.convert(q, src="quat", dst="quat")
    expected = SO3.canonicalize(q)

    assert converted.dtype == expected.dtype
    assert converted.shape == expected.shape
    assert np.allclose(np.array(converted), np.array(expected), atol=ATOL[precision])


@pytest.mark.parametrize("convention", ["bad", "xyz"])
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
def test_convert_rejects_unknown_quat_convention(convention, precision):
    q = random_quaternion(batch_dims=(4,), backend="numpy", precision=precision)

    with pytest.raises(ValueError, match="Unsupported quaternion convention"):
        SO3.convert(q, src="quat", dst="rotmat", src_convention=convention)


@pytest.mark.parametrize("rep", ["quat_wxyz", "quat_xyzw"])
def test_convert_rejects_legacy_quat_representation_strings(rep):
    with pytest.raises(ValueError, match="Unsupported rotation representation"):
        SO3.convert(np.zeros((4,)), src=rep, dst="rotmat")


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
    expected = SO3.conversions.from_rotmat_to_quat_wxyz(SO3.conversions.from_matrix_to_rotmat(matrix))

    dots = np.sum(np.array(converted) * np.array(expected), axis=-1)
    assert np.allclose(np.abs(dots), 1.0, atol=ATOL[32])


def test_convert_rejects_float16_matrix_projection():
    quat = random_quaternion(batch_dims=(4,), backend="numpy", precision=16)
    matrix = SO3.to_rotmat(quat)

    with pytest.raises(ValueError, match="does not support float16"):
        SO3.convert(matrix, src="matrix", dst="rotmat")


@pytest.mark.parametrize("src,dst", [("bad", "rotmat"), ("rotmat", "bad"), ("bad", "matrix"), ("matrix", "bad")])
def test_convert_rejects_unknown_representation(src, dst):
    with pytest.raises(ValueError, match="Unsupported rotation representation"):
        SO3.convert(np.zeros((3, 3)), src=src, dst=dst)
