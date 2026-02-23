"""Tests for SO3.conversions pairwise wrappers."""

import numpy as np
import pytest
from conftest import ATOL, TEST_BACKENDS, TEST_BATCH_DIMS, TEST_PASS_XP, get_xp_kwargs, random_quaternion

from nanomanifold import SO3

# ---------------------------------------------------------------------------
# Helpers to build inputs for each source representation
# ---------------------------------------------------------------------------

_EULER_CONVENTIONS = ["ZYX", "XYZ", "ZXZ"]


def _make_input(source: str, batch_dims, backend, precision=32, convention="ZYX"):
    """Return a sample array in the given representation."""
    q = random_quaternion(batch_dims=batch_dims, backend=backend, precision=precision)
    if source == "axis_angle":
        return SO3.to_axis_angle(q)
    if source == "euler":
        return SO3.to_euler(q, convention=convention)
    if source == "matrix":
        return SO3.to_matrix(q)
    if source == "quat_wxyz":
        return q
    if source == "quat_xyzw":
        return SO3.to_quat_xyzw(q)
    if source == "sixd":
        return SO3.to_6d(q)
    raise ValueError(source)


def _manual_convert(x, source, target, convention="ZYX"):
    """Convert via the two-step from/to primitive path (the reference)."""
    # source -> quat_wxyz
    if source == "axis_angle":
        q = SO3.from_axis_angle(x)
    elif source == "euler":
        q = SO3.from_euler(x, convention=convention)
    elif source == "matrix":
        q = SO3.from_matrix(x)
    elif source == "quat_wxyz":
        q = SO3.canonicalize(x)
    elif source == "quat_xyzw":
        q = SO3.from_quat_xyzw(x)
    elif source == "sixd":
        q = SO3.from_6d(x)
    else:
        raise ValueError(source)

    # quat_wxyz -> target
    if target == "axis_angle":
        return SO3.to_axis_angle(q)
    if target == "euler":
        return SO3.to_euler(q, convention=convention)
    if target == "matrix":
        return SO3.to_matrix(q)
    if target == "quat_wxyz":
        return SO3.canonicalize(q)
    if target == "quat_xyzw":
        return SO3.to_quat_xyzw(q)
    if target == "sixd":
        return SO3.to_6d(q)
    raise ValueError(target)


# ---------------------------------------------------------------------------
# All 30 source-target pairs
# ---------------------------------------------------------------------------

_REPS = ["axis_angle", "euler", "matrix", "quat_wxyz", "quat_xyzw", "sixd"]
_PAIRS = [(s, t) for s in _REPS for t in _REPS if s != t]


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
    x = _make_input(source, batch_dims, backend)

    fn = _get_conv_fn(source, target)
    result = fn(x, **xp_kwargs)
    expected = _manual_convert(x, source, target)

    result_np = np.array(result)
    expected_np = np.array(expected)

    atol = ATOL[32]
    if target in ("quat_wxyz", "quat_xyzw"):
        # quaternion sign ambiguity
        dots = np.sum(result_np * expected_np, axis=-1)
        assert np.allclose(np.abs(dots), 1.0, atol=atol), f"{source}->{target} quat mismatch"
    else:
        assert np.allclose(result_np, expected_np, atol=atol), f"{source}->{target} mismatch"


# ---------------------------------------------------------------------------
# 2. Round-trip: B_to_A(A_to_B(x)) ≈ x
# ---------------------------------------------------------------------------

_ROUND_TRIP_PAIRS = [
    ("axis_angle", "matrix"),
    ("axis_angle", "quat_wxyz"),
    ("axis_angle", "sixd"),
    ("matrix", "quat_wxyz"),
    ("matrix", "sixd"),
    ("quat_wxyz", "quat_xyzw"),
    ("quat_wxyz", "sixd"),
]


@pytest.mark.parametrize("pair", _ROUND_TRIP_PAIRS, ids=[f"{a}<->{b}" for a, b in _ROUND_TRIP_PAIRS])
@pytest.mark.parametrize("backend", TEST_BACKENDS)
@pytest.mark.parametrize("batch_dims", TEST_BATCH_DIMS)
def test_round_trip(pair, backend, batch_dims):
    a, b = pair
    x = _make_input(a, batch_dims, backend)

    a_to_b = _get_conv_fn(a, b)
    b_to_a = _get_conv_fn(b, a)

    recovered = b_to_a(a_to_b(x))

    x_np = np.array(x)
    rec_np = np.array(recovered)

    atol = ATOL[32]
    if a in ("quat_wxyz", "quat_xyzw"):
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
    result = SO3.conversions.from_euler_to_matrix(euler, convention=convention)
    expected = _manual_convert(euler, "euler", "matrix", convention=convention)
    assert np.allclose(np.array(result), np.array(expected), atol=ATOL[32])

    # euler as target
    aa = SO3.to_axis_angle(q)
    result2 = SO3.conversions.from_axis_angle_to_euler(aa, convention=convention)
    expected2 = _manual_convert(aa, "axis_angle", "euler", convention=convention)
    assert np.allclose(np.array(result2), np.array(expected2), atol=ATOL[32])
