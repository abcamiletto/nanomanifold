"""Tests for torch.compile(fullgraph=True) compatibility.

torch.compile with fullgraph=True requires that the entire forward pass can be
traced without graph breaks. Passing xp=torch skips the dynamic get_namespace()
call that Dynamo cannot trace through.
"""

import pytest

torch = pytest.importorskip("torch")

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
        return SO3.to_matrix(q, xp=torch)
    if rep == "quat_wxyz":
        return q
    if rep == "quat_xyzw":
        return SO3.to_quat_xyzw(q, xp=torch)
    if rep == "sixd":
        return SO3.to_6d(q, xp=torch)
    raise ValueError(rep)


_CONV_REPS = ["axis_angle", "euler", "matrix", "quat_wxyz", "quat_xyzw", "sixd"]
_CONV_PAIRS = [(s, t) for s in _CONV_REPS for t in _CONV_REPS if s != t]


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


def test_compile_from_matrix():
    def f(R):
        return SO3.from_matrix(R, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(torch.eye(3).unsqueeze(0).expand(2, -1, -1))


def test_compile_to_matrix():
    def f(q):
        return SO3.to_matrix(q, xp=torch)

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


def test_compile_from_6d():
    def f(d6):
        return SO3.from_6d(d6, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(torch.randn(2, 6))


def test_compile_to_6d():
    def f(q):
        return SO3.to_6d(q, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_random_quat())


@pytest.mark.parametrize("source,target", _CONV_PAIRS, ids=[f"{s}->{t}" for s, t in _CONV_PAIRS])
def test_compile_so3_pairwise_conversions(source, target):
    torch._dynamo.reset()
    fn = getattr(SO3.conversions, f"from_{source}_to_{target}")

    def f(x):
        return fn(x, xp=torch)

    compiled = torch.compile(f, fullgraph=True)
    compiled(_conv_input(source))


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


# ── SO3 composite pipelines ─────────────────────────────────────────────────


def test_compile_axis_angle_roundtrip():
    def f(x):
        return SO3.to_matrix(SO3.from_axis_angle(x, xp=torch), xp=torch)

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
