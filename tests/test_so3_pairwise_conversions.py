"""Tests for SO3.conversions pairwise wrappers."""

import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PASS_XP, get_xp_kwargs, random_quaternion

from nanomanifold import SO3
from nanomanifold.common import get_namespace_by_name

# ---------------------------------------------------------------------------
# Helpers to build inputs for each source representation
# ---------------------------------------------------------------------------

_EULER_CONVENTIONS = ["ZYX", "XYZ", "ZXZ"]
_QUAT_CONVENTION = "xyzw"


def _make_hinge_axes(backend):
    xp = get_namespace_by_name(backend)
    return xp.asarray(np.array([0.0, 0.0, 2.0], dtype=np.float32))


def _make_input(source: str, batch_dims, backend, precision=32, convention="ZYX", quat_convention="wxyz"):
    """Return a sample array in the given representation."""
    q = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)
    if source == "axis_angle":
        return SO3.to_axis_angle(q)
    if source == "euler":
        return SO3.to_euler(q, convention=convention)
    if source == "hinge":
        xp = get_namespace_by_name(backend)
        return xp.asarray(np.full(batch_dims + (1,), 0.2, dtype=f"float{precision}"))
    if source == "matrix":
        xp = get_namespace_by_name(backend)
        rotmat = SO3.to_rotmat(q)
        stretch = np.diag(np.array([1.05, 0.97, 1.02], dtype=np.float32))
        return xp.asarray(np.matmul(np.array(rotmat), stretch))
    if source == "rotmat":
        return SO3.to_rotmat(q)
    if source == "quat":
        return SO3.to_quat(q, convention=quat_convention)
    if source == "sixd":
        return SO3.to_sixd(q)
    raise ValueError(source)


def _manual_convert(x, source, target, axes=None, input_convention="ZYX", output_convention="ZYX"):
    """Convert via the two-step from/to primitive path (the reference)."""
    if source == "axis_angle":
        q = SO3.from_axis_angle(x)
    elif source == "euler":
        q = SO3.from_euler(x, convention=input_convention)
    elif source == "hinge":
        q = SO3.from_hinge(x, axes)
    elif source == "matrix":
        q = SO3.from_matrix(x)
    elif source == "rotmat":
        q = SO3.from_rotmat(x)
    elif source == "quat":
        q = SO3.from_quat(x, convention=input_convention)
    elif source == "sixd":
        q = SO3.from_sixd(x)
    else:
        raise ValueError(source)

    if target == "axis_angle":
        return SO3.to_axis_angle(q)
    if target == "euler":
        return SO3.to_euler(q, convention=output_convention)
    if target == "hinge":
        return SO3.to_hinge(q, axes)
    if target == "matrix":
        return SO3.to_rotmat(q)
    if target == "rotmat":
        return SO3.to_rotmat(q)
    if target == "quat":
        return SO3.to_quat(q, convention=output_convention)
    if target == "sixd":
        return SO3.to_sixd(q)
    raise ValueError(target)


# ---------------------------------------------------------------------------
# All supported source-target pairs
# ---------------------------------------------------------------------------

_SOURCE_REPS = ["axis_angle", "euler", "hinge", "matrix", "rotmat", "quat", "sixd"]
_TARGET_REPS = ["axis_angle", "euler", "hinge", "matrix", "rotmat", "quat", "sixd"]
_PAIRS = [(s, t) for s in _SOURCE_REPS for t in _TARGET_REPS if s != t]


def _get_conv_fn(source, target):
    """Look up SO3.conversions.from_<source>_to_<target>."""
    return getattr(SO3.conversions, f"from_{source}_to_{target}")


# ---------------------------------------------------------------------------
# 1. Equivalence: wrapper == manual two-step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pair", _PAIRS, ids=[f"{s}->{t}" for s, t in _PAIRS])
@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_equivalence(pair, backend, batch_dims, pass_xp):
    source, target = pair
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    input_convention = _QUAT_CONVENTION if source == "quat" else "XYZ"
    output_convention = _QUAT_CONVENTION if target == "quat" else "XYZ"
    axes = _make_hinge_axes(backend)
    x = _make_input(source, batch_dims, backend, convention="XYZ", quat_convention=input_convention)

    fn = _get_conv_fn(source, target)
    if source == "hinge" and target == "euler":
        result = fn(x, axes, convention=output_convention, **xp_kwargs)
    elif source == "hinge" and target == "quat":
        result = fn(x, axes, convention=output_convention, **xp_kwargs)
    elif source == "hinge":
        result = fn(x, axes, **xp_kwargs)
    elif target == "hinge" and source == "euler":
        result = fn(x, axes, convention=input_convention, **xp_kwargs)
    elif target == "hinge" and source == "quat":
        result = fn(x, axes, convention=input_convention, **xp_kwargs)
    elif target == "hinge":
        result = fn(x, axes, **xp_kwargs)
    elif source == "euler" and target == "euler":
        result = fn(x, input_convention=input_convention, output_convention=output_convention, **xp_kwargs)
    elif source == "euler" and target == "quat":
        result = fn(x, euler_convention=input_convention, quat_convention=output_convention, **xp_kwargs)
    elif source == "quat" and target == "euler":
        result = fn(x, quat_convention=input_convention, euler_convention=output_convention, **xp_kwargs)
    elif source == "quat" and target == "quat":
        result = fn(x, input_convention=input_convention, output_convention=output_convention, **xp_kwargs)
    elif source == "euler":
        result = fn(x, convention=input_convention, **xp_kwargs)
    elif target == "euler":
        result = fn(x, convention=output_convention, **xp_kwargs)
    elif target == "quat":
        result = fn(x, convention=output_convention, **xp_kwargs)
    elif source == "quat":
        result = fn(x, convention=input_convention, **xp_kwargs)
    else:
        result = fn(x, **xp_kwargs)
    expected = _manual_convert(x, source, target, axes=axes, input_convention=input_convention, output_convention=output_convention)

    result_np = np.array(result)
    expected_np = np.array(expected)

    atol = ATOL[32]
    if target == "quat":
        # quaternion sign ambiguity
        dots = np.sum(result_np * expected_np, axis=-1)
        assert np.allclose(np.abs(dots), 1.0, atol=atol), f"{source}->{target} quat mismatch"
    elif target == "euler":
        result_quat = SO3.from_euler(result, convention=output_convention)
        expected_quat = SO3.from_euler(expected, convention=output_convention)
        dots = np.sum(np.array(result_quat) * np.array(expected_quat), axis=-1)
        assert np.allclose(np.abs(dots), 1.0, atol=atol), f"{source}->{target} euler mismatch"
    else:
        assert np.allclose(result_np, expected_np, atol=atol), f"{source}->{target} mismatch"


# ---------------------------------------------------------------------------
# 2. Round-trip: B_to_A(A_to_B(x)) ≈ x
# ---------------------------------------------------------------------------

_ROUND_TRIP_PAIRS = [
    ("axis_angle", "rotmat"),
    ("axis_angle", "quat"),
    ("axis_angle", "sixd"),
    ("rotmat", "quat"),
    ("rotmat", "sixd"),
    ("quat", "sixd"),
]


@pytest.mark.parametrize("pair", _ROUND_TRIP_PAIRS, ids=[f"{a}<->{b}" for a, b in _ROUND_TRIP_PAIRS])
@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_round_trip(pair, backend, batch_dims):
    a, b = pair
    input_convention = _QUAT_CONVENTION if a == "quat" else "XYZ"
    output_convention = _QUAT_CONVENTION if b == "quat" else "XYZ"
    x = _make_input(a, batch_dims, backend, convention="XYZ", quat_convention=input_convention)

    a_to_b = _get_conv_fn(a, b)
    b_to_a = _get_conv_fn(b, a)
    if a == "euler" and b == "quat":
        converted = a_to_b(x, euler_convention=input_convention, quat_convention=output_convention)
        recovered = b_to_a(converted, quat_convention=output_convention, euler_convention=input_convention)
    elif a == "quat" and b == "euler":
        converted = a_to_b(x, quat_convention=input_convention, euler_convention=output_convention)
        recovered = b_to_a(converted, euler_convention=output_convention, quat_convention=input_convention)
    elif a == "euler":
        converted = a_to_b(x, convention=input_convention)
        recovered = b_to_a(converted, convention=input_convention)
    elif b == "euler":
        converted = a_to_b(x, convention=output_convention)
        recovered = b_to_a(converted, convention=output_convention)
    elif a == "quat":
        converted = a_to_b(x, convention=input_convention)
        recovered = b_to_a(converted, convention=input_convention)
    elif b == "quat":
        converted = a_to_b(x, convention=output_convention)
        recovered = b_to_a(converted, convention=output_convention)
    else:
        recovered = b_to_a(a_to_b(x))

    x_np = np.array(x)
    rec_np = np.array(recovered)

    atol = ATOL[32]
    if a == "quat":
        dots = np.sum(x_np * rec_np, axis=-1)
        assert np.allclose(np.abs(dots), 1.0, atol=atol), f"round-trip {a}<->{b} failed"
    else:
        assert np.allclose(x_np, rec_np, atol=atol), f"round-trip {a}<->{b} failed"


# ---------------------------------------------------------------------------
# 3. Euler convention threading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("convention", _EULER_CONVENTIONS)
@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_euler_convention_threading(convention, backend):
    q = random_quaternion(batch_dims=(5,), backend=backend)

    # euler as source
    euler = SO3.to_euler(q, convention=convention)
    result = SO3.conversions.from_euler_to_rotmat(euler, convention=convention)
    expected = _manual_convert(euler, "euler", "rotmat", input_convention=convention, output_convention=convention)
    assert np.allclose(np.array(result), np.array(expected), atol=ATOL[32])

    # euler as target
    aa = SO3.to_axis_angle(q)
    result2 = SO3.conversions.from_axis_angle_to_euler(aa, convention=convention)
    expected2 = _manual_convert(aa, "axis_angle", "euler", input_convention=convention, output_convention=convention)
    assert np.allclose(np.array(result2), np.array(expected2), atol=ATOL[32])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_from_euler_to_euler_rethreads_convention(backend, pass_xp):
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    q = random_quaternion(batch_dims=(5,), backend=backend)
    euler_xyz = SO3.to_euler(q, convention="XYZ")

    result = SO3.conversions.from_euler_to_euler(
        euler_xyz,
        input_convention="XYZ",
        output_convention="ZYX",
        **xp_kwargs,
    )
    expected = SO3.to_euler(SO3.from_euler(euler_xyz, convention="XYZ"), convention="ZYX")

    assert result.dtype == expected.dtype
    assert result.shape == expected.shape
    assert np.allclose(np.array(result), np.array(expected), atol=ATOL[32])


@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("pass_xp", TEST_PASS_XP)
def test_from_quat_to_quat_rethreads_convention(backend, pass_xp):
    xp_kwargs = get_xp_kwargs(backend, pass_xp)
    q = random_quaternion(batch_dims=(5,), backend=backend)
    quat_xyzw = SO3.to_quat(q, convention="xyzw")

    result = SO3.conversions.from_quat_to_quat(
        quat_xyzw,
        input_convention="xyzw",
        output_convention="wxyz",
        **xp_kwargs,
    )
    expected = SO3.canonicalize(q)

    assert result.dtype == expected.dtype
    assert result.shape == expected.shape
    assert np.allclose(np.array(result), np.array(expected), atol=ATOL[32])
