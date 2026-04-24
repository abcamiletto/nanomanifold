"""Tests for torch.compile(fullgraph=True) compatibility.

torch.compile with fullgraph=True requires that the entire forward pass can be
traced without graph breaks. Passing xp=torch skips the dynamic get_namespace()
call that Dynamo cannot trace through.
"""

import pytest

torch = pytest.importorskip("torch")
if int(torch.__version__.split(".", 1)[0]) < 2 or not hasattr(torch, "compile"):
    pytest.skip("torch.compile tests require torch 2.x.", allow_module_level=True)

import nanomanifold.common as common  # noqa: E402
from nanomanifold import SE3, SO3  # noqa: E402


def _random_quat(batch_size=2):
    q = torch.randn(batch_size, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    q = torch.where(q[..., :1] < 0, -q, q)
    return q


def _conv_input(rep: str):
    q = _random_quat()
    if rep == "axis_angle":
        return SO3.to_axis_angle(q, xp=torch)
    if rep == "euler":
        return SO3.to_euler(q, "ZYX", xp=torch)
    if rep == "matrix":
        return SO3.to_rotmat(q, xp=torch) @ torch.diag(torch.tensor([1.05, 0.97, 1.02]))
    if rep == "rotmat":
        return SO3.to_rotmat(q, xp=torch)
    if rep == "quat":
        return SO3.to_quat(q, xp=torch)
    if rep == "sixd":
        return SO3.to_sixd(q, xp=torch)
    if rep == "hinge":
        return torch.tensor([[0.1], [-0.2]])
    raise ValueError(rep)


_CONV_REPS = ["axis_angle", "euler", "hinge", "matrix", "rotmat", "quat", "sixd"]
_PAIRWISE_SOURCE_REPS = ["axis_angle", "euler", "hinge", "matrix", "rotmat", "quat", "sixd"]
_PAIRWISE_TARGET_REPS = ["axis_angle", "euler", "hinge", "rotmat", "quat", "sixd"]
_CONV_PAIRS = [(s, t) for s in _PAIRWISE_SOURCE_REPS for t in _PAIRWISE_TARGET_REPS if s != t]
_DYNAMIC_CONV_CASES = [
    ("axis_angle", "rotmat"),
    ("matrix", "rotmat"),
    ("euler", "quat"),
    ("rotmat", "euler"),
    ("quat", "sixd"),
    ("quat", "quat"),
    ("sixd", "axis_angle"),
    ("euler", "euler"),
    ("hinge", "axis_angle"),
    ("rotmat", "hinge"),
]


def _pairwise_input(rep: str):
    q = _random_quat()
    if rep == "axis_angle":
        return SO3.to_axis_angle(q, xp=torch)
    if rep == "euler":
        return SO3.to_euler(q, "ZYX", xp=torch)
    if rep == "matrix":
        return SO3.to_rotmat(q, xp=torch) @ torch.diag(torch.tensor([1.05, 0.97, 1.02]))
    if rep == "rotmat":
        return SO3.to_rotmat(q, xp=torch)
    if rep == "quat":
        return SO3.to_quat(q, convention="xyzw", xp=torch)
    if rep == "sixd":
        return SO3.to_sixd(q, xp=torch)
    if rep == "hinge":
        return torch.tensor([[0.1], [-0.2]])
    raise ValueError(rep)


# ── SO3 conversions ──────────────────────────────────────────────────────────


def test_compile_from_axis_angle():
    def f(x):
        return SO3.from_axis_angle(x, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(torch.randn(2, 3))


def test_compile_to_axis_angle():
    def f(q):
        return SO3.to_axis_angle(q, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_quat())


def test_compile_from_hinge():
    def f(angles, axes):
        return SO3.from_hinge(angles, axes, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(torch.tensor([[0.1], [-0.2]]), torch.tensor([0.0, 0.0, 1.0]))


def test_compile_to_hinge():
    axes = torch.tensor([0.0, 0.0, 1.0])

    def f(q):
        return SO3.to_hinge(q, axes, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_quat())


def test_compile_from_rotmat():
    def f(R):
        return SO3.from_rotmat(R, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(torch.eye(3).unsqueeze(0).expand(2, -1, -1))


def test_compile_from_matrix_normalize():
    def f(R):
        return SO3.from_matrix(R, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(torch.eye(3).unsqueeze(0).expand(2, -1, -1))


def test_compile_from_matrix_normalize_davenport():
    def f(R):
        return SO3.from_matrix(R, mode="davenport", xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(torch.eye(3).unsqueeze(0).expand(2, -1, -1))


def test_compile_to_rotmat():
    def f(q):
        return SO3.to_rotmat(q, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_quat())


def test_compile_from_euler():
    def f(e):
        return SO3.from_euler(e, "XYZ", xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(torch.randn(2, 3))


def test_compile_to_euler():
    def f(q):
        return SO3.to_euler(q, "XYZ", xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_quat())


def test_compile_from_euler_to_euler():
    def f(e):
        return SO3.conversions.from_euler_to_euler(e, src_convention="XYZ", dst_convention="ZYX", xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(torch.randn(2, 3))


def test_compile_from_sixd():
    def f(sixd):
        return SO3.from_sixd(sixd, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(torch.randn(2, 6))


def test_compile_to_sixd():
    def f(q):
        return SO3.to_sixd(q, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_quat())


@pytest.mark.parametrize("source,target", _CONV_PAIRS, ids=[f"{s}->{t}" for s, t in _CONV_PAIRS])
def test_compile_so3_pairwise_conversions(source, target):
    torch._dynamo.reset()
    fn = getattr(SO3.conversions, f"from_{source}_to_{target}")
    axes = torch.tensor([0.0, 0.0, 1.0])

    if source == "hinge" and target == "euler":

        def f(x):
            return fn(x, axes, convention="XYZ", xp=torch)
    elif source == "hinge" and target == "quat":

        def f(x):
            return fn(x, axes, convention="xyzw", xp=torch)
    elif source == "hinge":

        def f(x):
            return fn(x, axes, xp=torch)
    elif target == "hinge" and source == "euler":

        def f(x):
            return fn(x, axes, convention="XYZ", xp=torch)
    elif target == "hinge" and source == "quat":

        def f(x):
            return fn(x, axes, convention="xyzw", xp=torch)
    elif target == "hinge":

        def f(x):
            return fn(x, axes, xp=torch)
    elif source == "euler" and target == "quat":

        def f(x):
            return fn(x, src_convention="XYZ", dst_convention="xyzw", xp=torch)
    elif source == "quat" and target == "euler":

        def f(x):
            return fn(x, src_convention="xyzw", dst_convention="XYZ", xp=torch)
    elif source == "quat" and target == "quat":

        def f(x):
            return fn(x, src_convention="xyzw", dst_convention="wxyz", xp=torch)
    elif source == "euler":

        def f(x):
            return fn(x, convention="XYZ", xp=torch)
    elif target == "euler":

        def f(x):
            return fn(x, convention="XYZ", xp=torch)
    elif source == "quat":

        def f(x):
            return fn(x, convention="xyzw", xp=torch)
    elif target == "quat":

        def f(x):
            return fn(x, convention="xyzw", xp=torch)
    else:

        def f(x):
            return fn(x, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_pairwise_input(source))


@pytest.mark.parametrize("source,target", _DYNAMIC_CONV_CASES, ids=[f"{s}->{t}" for s, t in _DYNAMIC_CONV_CASES])
def test_compile_so3_convert(source, target):
    torch._dynamo.reset()
    source_input = _conv_input(source)
    axes = torch.tensor([0.0, 0.0, 1.0])
    kwargs = {}
    if source == "hinge" and target != "hinge":
        kwargs["src_kwargs"] = {"axes": axes}
    if target == "hinge" and source != "hinge":
        kwargs["dst_kwargs"] = {"axes": axes}

    def f(x):
        return SO3.convert(
            x,
            src=source,
            dst=target,
            **kwargs,
            xp=torch,
        )

    compiled = torch.compile(f, fullgraph=True)
    compiled(source_input)


# ── SO3 operations ───────────────────────────────────────────────────────────


def test_compile_multiply():
    def f(q1, q2):
        return SO3.multiply(q1, q2, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_quat(), _random_quat())


def test_compile_inverse():
    def f(q):
        return SO3.inverse(q, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_quat())


def test_compile_rotate_points():
    def f(q, pts):
        return SO3.rotate_points(q, pts, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_quat(), torch.randn(2, 5, 3))


def test_compile_distance():
    def f(q1, q2):
        return SO3.distance(q1, q2, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_quat(), _random_quat())


def test_compile_distance_rotmat():
    def f(r1, r2):
        return SO3.distance(r1, r2, rotation_type="rotmat", xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    rotmat = torch.eye(3).unsqueeze(0).expand(2, -1, -1)
    compiled(rotmat, rotmat)


def test_compile_distance_euler_convention():
    def f(e1, e2):
        return SO3.distance(e1, e2, rotation_type="euler", convention="XYZ", xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    euler = torch.zeros(2, 3)
    compiled(euler, euler)


def test_compile_exp():
    def f(v):
        return SO3.exp(v, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(torch.randn(2, 3))


def test_compile_log():
    def f(q):
        return SO3.log(q, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_quat())


def test_compile_hat():
    def f(w):
        return SO3.hat(w, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(torch.randn(2, 3))


def test_compile_vee():
    def f(W):
        return SO3.vee(W, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    W = torch.zeros(2, 3, 3)
    W[:, 0, 1] = -1.0
    W[:, 1, 0] = 1.0
    compiled(W)


def test_compile_slerp():
    def f(q1, q2, t):
        return SO3.slerp(q1, q2, t, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_quat(), _random_quat(), torch.tensor(0.5))


def test_compile_canonicalize():
    def f(q):
        return SO3.canonicalize(q, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_quat())


def test_compile_zeros_as():
    def f(q):
        return common.zeros_as(q, shape=q.shape, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_quat())


def test_compile_identity_as():
    def f(q):
        return SO3.identity_as(q, batch_dims=q.shape[:-1], rotation_type="rotmat", xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_quat())


# ── SO3 composite pipelines ─────────────────────────────────────────────────


def test_compile_axis_angle_roundtrip():
    def f(x):
        return SO3.to_rotmat(SO3.from_axis_angle(x, xp=torch), xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(torch.randn(2, 3))


# ── SE3 ──────────────────────────────────────────────────────────────────────


def _random_se3(batch_size=2):
    q = _random_quat(batch_size)
    t = torch.randn(batch_size, 3)
    return torch.cat([q, t], dim=-1)


def test_compile_se3_multiply():
    def f(a, b):
        return SE3.multiply(a, b, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_se3(), _random_se3())


def test_compile_se3_inverse():
    def f(se3):
        return SE3.inverse(se3, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_se3())


def test_compile_se3_to_matrix():
    def f(se3):
        return SE3.to_matrix(se3, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_se3())


def test_compile_se3_from_rt():
    def f(q, t):
        return SE3.from_rt(q, t, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_quat(), torch.randn(2, 3))


def test_compile_se3_transform_points():
    def f(se3, pts):
        return SE3.transform_points(se3, pts, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_se3(), torch.randn(2, 5, 3))


def test_compile_se3_log():
    def f(se3):
        return SE3.log(se3, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_se3())


def test_compile_se3_exp():
    def f(v):
        return SE3.exp(v, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(torch.randn(2, 6))


def test_compile_se3_canonicalize():
    def f(se3):
        return SE3.canonicalize(se3, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_se3())


def test_compile_se3_from_matrix():
    def f(M):
        return SE3.from_matrix(M, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    M = torch.eye(4).unsqueeze(0).expand(2, -1, -1).clone()
    M[:, :3, 3] = torch.randn(2, 3)
    compiled(M)


def test_compile_se3_slerp():
    def f(a, b, t):
        return SE3.slerp(a, b, t, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_se3(), _random_se3(), torch.tensor(0.5))


def test_compile_se3_hat():
    def f(v):
        return SE3.hat(v, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(torch.randn(2, 6))


def test_compile_so3_random():
    def f():
        return SO3.random(2, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled()


def test_compile_se3_random():
    def f():
        return SE3.random(2, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled()


def test_compile_se3_vee():
    def f(M):
        return SE3.vee(M, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    M = torch.zeros(2, 4, 4)
    M[:, 0, 1] = -1.0
    M[:, 1, 0] = 1.0
    M[:, :3, 3] = torch.randn(2, 3)
    compiled(M)
