"""Every public function on torch and JAX.

Each case runs on the awkward rotation batch from conftest and must agree with NumPy, keep the input dtype and
shape, trace under ``torch.compile(fullgraph=True)`` and ``jax.jit``, pass ``torch.autograd.gradcheck`` on regular
inputs and keep gradients finite on the awkward ones, on torch and JAX. The math itself is tested on NumPy in the other files.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from conftest import awkward_quaternions, without_singularities

from nanomanifold import SE3, SO3
from nanomanifold.common import get_namespace_by_name

AXES = np.array([0.0, 0.0, 2.0])
ATOL = {16: 5e-2, 32: 1e-5}
REPS = ["axis_angle", "euler", "hinge", "matrix", "rotmat", "quat", "sixd"]
SRC_OPTIONS = {"quat": {"convention": "xyzw"}, "euler": {"convention": "XYZ"}, "matrix": {"mode": "davenport"}}
DST_OPTIONS = {"quat": {"convention": "xyzw"}, "euler": {"convention": "XYZ"}}


@dataclass(frozen=True)
class Case:
    name: str
    fn: Callable  # fn(*arrays, xp=...)
    inputs: Callable[[np.ndarray], tuple]  # canonical float64 quaternions -> float64 input arrays
    # Maps an output to a representation-independent form, e.g. a rotation matrix for a quaternion.
    canonical: Callable = lambda out: out
    precisions: tuple = (16, 32)
    differentiable: bool = True


def _rng():
    return np.random.default_rng(0)


def _pairs(q):
    """Identical, sign-flipped and 1 rad apart rotations.

    Rotations half a turn apart are left out: they have two shortest paths between them.
    """
    nearby = SO3.multiply(q, SO3.from_axis_angle(np.array([0.6, -0.48, 0.64])))
    return np.concatenate([q, q, q]), np.concatenate([q, -q, nearby])


def _points(q):
    return _rng().normal(size=(len(q), 5, 3))


def _t(q):
    return np.broadcast_to(np.array([0.0, 0.3, 1.0]), (len(q), 3)).copy()


def _se3(q):
    return np.concatenate([q, _rng().normal(size=(len(q), 3))], axis=-1)


def _random_quats(n):
    return SO3.canonicalize(_rng().normal(size=(n, 4)))


def _rotmat(q):
    return SO3.to_rotmat(q)


def _se3_matrix(T):
    return SE3.to_matrix(T)


def _options(options, rep, axes=None):
    return {"axes": axes, **options.get(rep, {})} if rep == "hinge" else options.get(rep, {})


def _convert_case(src, dst):
    uses_axes = "hinge" in (src, dst)

    def fn(x, *axes, xp):
        src_kwargs = _options(SRC_OPTIONS, src, *axes)
        dst_kwargs = _options(DST_OPTIONS, dst, *axes)
        return SO3.convert(x, src=src, dst=dst, src_kwargs=src_kwargs, dst_kwargs=dst_kwargs, xp=xp)

    def inputs(q):
        q = without_singularities(without_singularities(q, src, SRC_OPTIONS.get(src, {})), dst, DST_OPTIONS.get(dst, {}))
        # A rotation matrix is a valid generic matrix; Davenport needs no stretch to be exercised.
        rep = "rotmat" if src == "matrix" else src
        x = SO3.convert(q, src="quat", dst=rep, dst_kwargs=_options(SRC_OPTIONS, rep, AXES))
        return (x, AXES) if uses_axes else (x,)

    def canonical(out):
        return SO3.convert(out, src=dst, dst="rotmat", src_kwargs=_options(DST_OPTIONS, dst, AXES))

    return Case(f"convert-{src}-{dst}", fn, inputs, canonical)


CASES = [
    # Routes are backend-independent Python; going to and from quat crosses every edge of the tree in both directions.
    *[_convert_case("quat", rep) for rep in REPS],
    *[_convert_case(rep, "quat") for rep in REPS if rep != "quat"],
    Case(
        "convert-matrix-svd",
        lambda m, xp: SO3.convert(m, src="matrix", dst="quat", src_kwargs={"mode": "svd"}, xp=xp),
        lambda q: (_rotmat(q) @ np.diag([1.05, 0.97, 1.02]),),
        _rotmat,
        precisions=(32,),
    ),
    Case(
        "SO3.canonicalize",
        lambda q, xp: SO3.canonicalize(q, convention="xyzw", xp=xp),
        lambda q: (-2 * q,),
        lambda q: SO3.to_rotmat(q, convention="xyzw"),
    ),
    Case("SO3.from_quat", lambda q, xp: SO3.from_quat(q, convention="xyzw", xp=xp), lambda q: (q,), _rotmat),
    Case(
        "SO3.to_quat", lambda q, xp: SO3.to_quat(q, convention="xyzw", xp=xp), lambda q: (q,), lambda q: SO3.to_rotmat(q, convention="xyzw")
    ),
    Case("SO3.multiply", lambda a, b, xp: SO3.multiply(a, b, xp=xp), _pairs, _rotmat),
    Case("SO3.inverse", lambda q, xp: SO3.inverse(q, xp=xp), lambda q: (q,), _rotmat),
    Case("SO3.rotate_points", lambda q, p, xp: SO3.rotate_points(q, p, xp=xp), lambda q: (q, _points(q))),
    Case("SO3.slerp", lambda a, b, t, xp: SO3.slerp(a, b, t, xp=xp), lambda q: (*_pairs(q), _t(np.concatenate([q, q, q]))), _rotmat),
    Case("SO3.distance", lambda a, b, xp: SO3.distance(a, b, xp=xp), _pairs),
    Case("SO3.log", lambda q, xp: SO3.log(q, xp=xp), lambda q: (q,), lambda w: SO3.to_rotmat(SO3.exp(w))),
    Case("SO3.exp", lambda w, xp: SO3.exp(w, xp=xp), lambda q: (SO3.log(q),), _rotmat),
    Case("SO3.hat", lambda w, xp: SO3.hat(w, xp=xp), lambda q: (SO3.log(q),)),
    Case("SO3.vee", lambda m, xp: SO3.vee(m, xp=xp), lambda q: (SO3.hat(SO3.log(q)),)),
    Case(
        "SO3.weighted_mean",
        lambda a, b, w, xp: SO3.weighted_mean([a, b], w, xp=xp),
        lambda q: (q, np.roll(q, 1, axis=0), np.broadcast_to([0.7, 0.3], (len(q), 2)).copy()),
        _rotmat,
    ),
    Case("SO3.mean", lambda a, b, c, xp: SO3.mean([a, b, c], xp=xp), lambda q: tuple(_random_quats(8) for _ in range(3)), _rotmat),
    Case(
        "SO3.identity_as",
        lambda ref, xp: SO3.identity_as(ref, batch_dims=ref.shape[:-1], rep="rotmat", xp=xp),
        lambda q: (q,),
        differentiable=False,
    ),
    Case(
        "SE3.identity_as",
        lambda ref, xp: SE3.identity_as(ref, batch_dims=ref.shape[:-1], convention="xyzw", xp=xp),
        lambda q: (_se3(q),),
        differentiable=False,
    ),
    Case("SE3.canonicalize", lambda T, xp: SE3.canonicalize(T, xp=xp), lambda q: (_se3(-q),), _se3_matrix),
    Case("SE3.from_rt", lambda q, t, xp: SE3.from_rt(q, t, xp=xp), lambda q: (q, _se3(q)[..., 4:]), _se3_matrix),
    Case(
        "SE3.to_rt",
        lambda T, xp: SE3.to_rt(T, xp=xp),
        lambda q: (_se3(q),),
        lambda out: np.concatenate([_rotmat(out[0]).reshape(-1, 9), out[1]], axis=-1),
    ),
    Case("SE3.from_matrix", lambda m, xp: SE3.from_matrix(m, xp=xp), lambda q: (_se3_matrix(_se3(q)),), _se3_matrix),
    Case("SE3.to_matrix", lambda T, xp: SE3.to_matrix(T, xp=xp), lambda q: (_se3(q),)),
    Case("SE3.multiply", lambda a, b, xp: SE3.multiply(a, b, xp=xp), lambda q: tuple(_se3(x) for x in _pairs(q)), _se3_matrix),
    Case("SE3.inverse", lambda T, xp: SE3.inverse(T, xp=xp), lambda q: (_se3(q),), _se3_matrix),
    Case("SE3.transform_points", lambda T, p, xp: SE3.transform_points(T, p, xp=xp), lambda q: (_se3(q), _points(q))),
    Case(
        "SE3.slerp",
        lambda a, b, t, xp: SE3.slerp(a, b, t, xp=xp),
        lambda q: (*(_se3(x) for x in _pairs(q)), _t(np.concatenate([q, q, q]))),
        _se3_matrix,
    ),
    Case("SE3.log", lambda T, xp: SE3.log(T, xp=xp), lambda q: (_se3(q),), lambda xi: _se3_matrix(SE3.exp(xi))),
    Case("SE3.exp", lambda xi, xp: SE3.exp(xi, xp=xp), lambda q: (SE3.log(_se3(q)),), _se3_matrix),
    Case("SE3.hat", lambda xi, xp: SE3.hat(xi, xp=xp), lambda q: (SE3.log(_se3(q)),)),
    Case("SE3.vee", lambda m, xp: SE3.vee(m, xp=xp), lambda q: (SE3.hat(SE3.log(_se3(q))),)),
    Case(
        "SE3.weighted_mean",
        lambda a, b, w, xp: SE3.weighted_mean([a, b], w, xp=xp),
        lambda q: (_se3(q), _se3(np.roll(q, 1, axis=0)), np.broadcast_to([0.7, 0.3], (len(q), 2)).copy()),
        _se3_matrix,
    ),
    Case(
        "SE3.mean", lambda a, b, c, xp: SE3.mean([a, b, c], xp=xp), lambda q: tuple(_se3(_random_quats(8)) for _ in range(3)), _se3_matrix
    ),
]
CASE_IDS = [case.name for case in CASES]
DIFFERENTIABLE = [case for case in CASES if case.differentiable]


def _to_numpy(out):
    return tuple(np.asarray(x, dtype=np.float64) for x in out) if isinstance(out, tuple) else np.asarray(out, dtype=np.float64)


def _torch_compile():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile requires torch 2.")
    torch._dynamo.reset()
    return torch


def _flatten(out):
    return out if isinstance(out, tuple) else (out,)


def _assert_same(case, result, expected, atol):
    for res, exp in zip(_flatten(case.canonical(_to_numpy(result))), _flatten(case.canonical(_to_numpy(expected)))):
        assert np.allclose(res, exp, atol=atol)


def _inputs(case, precision, xp, quats=None):
    arrays = case.inputs(awkward_quaternions() if quats is None else quats)
    return tuple(np.asarray(a, dtype=f"float{precision}") for a in arrays), tuple(
        xp.asarray(np.asarray(a, dtype=f"float{precision}")) for a in arrays
    )


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("precision", [16, 32])
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_matches_numpy(case, precision, backend):
    if precision not in case.precisions:
        pytest.skip(f"{case.name} does not support float{precision}.")
    xp = get_namespace_by_name(backend)
    numpy_inputs, backend_inputs = _inputs(case, precision, xp)

    expected = case.fn(*numpy_inputs, xp=np)
    result = case.fn(*backend_inputs, xp=None)  # infer the backend from the inputs

    for res, exp in zip(_flatten(result), _flatten(expected)):
        assert type(res).__module__.split(".")[0] == type(backend_inputs[0]).__module__.split(".")[0]
        assert np.asarray(res).dtype == exp.dtype
        assert res.shape == exp.shape
    _assert_same(case, result, expected, ATOL[precision])


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_torch_compile(case):
    torch = _torch_compile()
    _, inputs = _inputs(case, 32, torch)

    compiled = torch.compile(lambda *args: case.fn(*args, xp=torch), fullgraph=True)

    _assert_same(case, compiled(*inputs), case.fn(*inputs, xp=torch), ATOL[32])


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_jax_jit(case):
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    _, inputs = _inputs(case, 32, jnp)

    jitted = jax.jit(lambda *args: case.fn(*args, xp=jnp))

    _assert_same(case, jitted(*inputs), case.fn(*inputs, xp=jnp), ATOL[32])


@pytest.mark.parametrize("case", DIFFERENTIABLE, ids=[case.name for case in DIFFERENTIABLE])
def test_torch_gradcheck(case):
    torch = pytest.importorskip("torch")
    _, inputs = _inputs(case, 64, torch, quats=_random_quats(4))
    inputs = tuple(x.requires_grad_(True) for x in inputs)

    assert torch.autograd.gradcheck(lambda *args: _flatten(case.fn(*args, xp=torch)), inputs, eps=1e-6, atol=1e-5)


def _awkward_gradients(case, backend):
    """Gradients of the summed outputs with respect to every input, in float64 on the awkward batch."""
    if backend == "torch":
        torch = pytest.importorskip("torch")
        _, inputs = _inputs(case, 64, torch)
        inputs = tuple(x.requires_grad_(True) for x in inputs)
        sum(out.sum() for out in _flatten(case.fn(*inputs, xp=torch))).backward()
        return [x.grad.numpy() for x in inputs]

    jax = pytest.importorskip("jax")
    # jax >= 0.5 scopes float64 with jax.enable_x64, older versions with jax.experimental.enable_x64.
    with jax.enable_x64(True) if hasattr(jax, "enable_x64") else jax.experimental.enable_x64():
        _, inputs = _inputs(case, 64, jax.numpy)
        loss = lambda *args: sum(out.sum() for out in _flatten(case.fn(*args, xp=jax.numpy)))  # noqa: E731
        return [np.asarray(g) for g in jax.grad(loss, argnums=tuple(range(len(inputs))))(*inputs)]


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("case", DIFFERENTIABLE, ids=[case.name for case in DIFFERENTIABLE])
def test_gradients_finite_on_awkward_inputs(case, backend):
    for grad in _awkward_gradients(case, backend):
        assert np.isfinite(grad).all()


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("precision", [16, 32])
@pytest.mark.parametrize("group,size", [(SO3, 4), (SE3, 7)], ids=["SO3", "SE3"])
def test_random(group, size, precision, backend):
    xp = get_namespace_by_name(backend)
    key = pytest.importorskip("jax").random.PRNGKey(0) if backend == "jax" else None

    sample = group.random(3, 2, dtype=getattr(xp, f"float{precision}"), key=key, xp=xp)

    assert sample.shape == (3, 2, size)
    assert sample.dtype == getattr(xp, f"float{precision}")
    assert np.allclose(np.linalg.norm(np.asarray(sample[..., :4], dtype=np.float64), axis=-1), 1.0, atol=ATOL[precision])


@pytest.mark.parametrize("group", [SO3, SE3], ids=["SO3", "SE3"])
def test_random_jax_requires_key_and_is_deterministic_under_jit(group):
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    key = jax.random.PRNGKey(0)

    with pytest.raises(ValueError, match="PRNG key"):
        group.random(5, xp=jnp)
    sample = jax.jit(lambda key: group.random(5, key=key, xp=jnp))

    assert np.allclose(sample(key), group.random(5, key=key, xp=jnp), atol=1e-6)


@pytest.mark.parametrize("group", [SO3, SE3], ids=["SO3", "SE3"])
def test_random_torch_compile(group):
    torch = _torch_compile()

    sample = torch.compile(lambda: group.random(5, xp=torch), fullgraph=True)()

    assert sample.shape == (5, 4 if group is SO3 else 7)
