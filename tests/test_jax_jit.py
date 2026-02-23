"""Tests for jax.jit compatibility.

Mirrors test_torch_compile.py: every SO3/SE3 function must be traceable by
jax.jit without errors.  Passing xp=jnp avoids the dynamic get_namespace()
call that the JAX tracer cannot see through.
"""

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from nanomanifold import SE3, SO3  # noqa: E402


def _random_quat(batch_size=2):
    q = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 4))
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    q = jnp.where(q[..., :1] < 0, -q, q)
    return q


# ── SO3 conversions ──────────────────────────────────────────────────────────


def test_jit_from_axis_angle():
    compiled = jax.jit(lambda x: SO3.from_axis_angle(x, xp=jnp))
    compiled(jax.random.normal(jax.random.PRNGKey(0), (2, 3)))


def test_jit_to_axis_angle():
    compiled = jax.jit(lambda q: SO3.to_axis_angle(q, xp=jnp))
    compiled(_random_quat())


def test_jit_from_matrix():
    compiled = jax.jit(lambda R: SO3.from_matrix(R, xp=jnp))
    compiled(jnp.broadcast_to(jnp.eye(3), (2, 3, 3)))


def test_jit_to_matrix():
    compiled = jax.jit(lambda q: SO3.to_matrix(q, xp=jnp))
    compiled(_random_quat())


def test_jit_from_euler():
    compiled = jax.jit(lambda e: SO3.from_euler(e, "XYZ", xp=jnp))
    compiled(jax.random.normal(jax.random.PRNGKey(0), (2, 3)))


def test_jit_to_euler():
    compiled = jax.jit(lambda q: SO3.to_euler(q, "XYZ", xp=jnp))
    compiled(_random_quat())


def test_jit_from_6d():
    compiled = jax.jit(lambda d6: SO3.from_6d(d6, xp=jnp))
    compiled(jax.random.normal(jax.random.PRNGKey(0), (2, 6)))


def test_jit_to_6d():
    compiled = jax.jit(lambda q: SO3.to_6d(q, xp=jnp))
    compiled(_random_quat())


# ── SO3 operations ───────────────────────────────────────────────────────────


def test_jit_multiply():
    compiled = jax.jit(lambda q1, q2: SO3.multiply(q1, q2, xp=jnp))
    compiled(_random_quat(), _random_quat())


def test_jit_inverse():
    compiled = jax.jit(lambda q: SO3.inverse(q, xp=jnp))
    compiled(_random_quat())


def test_jit_rotate_points():
    compiled = jax.jit(lambda q, pts: SO3.rotate_points(q, pts, xp=jnp))
    compiled(_random_quat(), jax.random.normal(jax.random.PRNGKey(0), (2, 5, 3)))


def test_jit_distance():
    compiled = jax.jit(lambda q1, q2: SO3.distance(q1, q2, xp=jnp))
    compiled(_random_quat(), _random_quat())


def test_jit_exp():
    compiled = jax.jit(lambda v: SO3.exp(v, xp=jnp))
    compiled(jax.random.normal(jax.random.PRNGKey(0), (2, 3)))


def test_jit_log():
    compiled = jax.jit(lambda q: SO3.log(q, xp=jnp))
    compiled(_random_quat())


def test_jit_hat():
    compiled = jax.jit(lambda w: SO3.hat(w, xp=jnp))
    compiled(jax.random.normal(jax.random.PRNGKey(0), (2, 3)))


def test_jit_vee():
    compiled = jax.jit(lambda W: SO3.vee(W, xp=jnp))
    W = jnp.zeros((2, 3, 3)).at[:, 0, 1].set(-1.0).at[:, 1, 0].set(1.0)
    compiled(W)


def test_jit_slerp():
    compiled = jax.jit(lambda q1, q2, t: SO3.slerp(q1, q2, t, xp=jnp))
    compiled(_random_quat(), _random_quat(), jnp.array(0.5))


def test_jit_canonicalize():
    compiled = jax.jit(lambda q: SO3.canonicalize(q, xp=jnp))
    compiled(_random_quat())


# ── SO3 composite pipelines ─────────────────────────────────────────────────


def test_jit_axis_angle_roundtrip():
    compiled = jax.jit(lambda x: SO3.to_matrix(SO3.from_axis_angle(x, xp=jnp), xp=jnp))
    compiled(jax.random.normal(jax.random.PRNGKey(0), (2, 3)))


# ── SE3 ──────────────────────────────────────────────────────────────────────


def _random_se3(batch_size=2):
    q = _random_quat(batch_size)
    t = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 3))
    return jnp.concatenate([q, t], axis=-1)


def test_jit_se3_multiply():
    compiled = jax.jit(lambda a, b: SE3.multiply(a, b, xp=jnp))
    compiled(_random_se3(), _random_se3())


def test_jit_se3_inverse():
    compiled = jax.jit(lambda se3: SE3.inverse(se3, xp=jnp))
    compiled(_random_se3())


def test_jit_se3_to_matrix():
    compiled = jax.jit(lambda se3: SE3.to_matrix(se3, xp=jnp))
    compiled(_random_se3())


def test_jit_se3_from_rt():
    compiled = jax.jit(lambda q, t: SE3.from_rt(q, t, xp=jnp))
    compiled(_random_quat(), jax.random.normal(jax.random.PRNGKey(0), (2, 3)))


def test_jit_se3_transform_points():
    compiled = jax.jit(lambda se3, pts: SE3.transform_points(se3, pts, xp=jnp))
    compiled(_random_se3(), jax.random.normal(jax.random.PRNGKey(0), (2, 5, 3)))


def test_jit_se3_log():
    compiled = jax.jit(lambda se3: SE3.log(se3, xp=jnp))
    compiled(_random_se3())


def test_jit_se3_exp():
    compiled = jax.jit(lambda v: SE3.exp(v, xp=jnp))
    compiled(jax.random.normal(jax.random.PRNGKey(0), (2, 6)))


def test_jit_se3_canonicalize():
    compiled = jax.jit(lambda se3: SE3.canonicalize(se3, xp=jnp))
    compiled(_random_se3())


def test_jit_se3_from_matrix():
    compiled = jax.jit(lambda M: SE3.from_matrix(M, xp=jnp))
    M = jnp.broadcast_to(jnp.eye(4), (2, 4, 4)).at[:, :3, 3].set(jax.random.normal(jax.random.PRNGKey(0), (2, 3)))
    compiled(M)


def test_jit_se3_slerp():
    compiled = jax.jit(lambda a, b, t: SE3.slerp(a, b, t, xp=jnp))
    compiled(_random_se3(), _random_se3(), jnp.array(0.5))


def test_jit_se3_hat():
    compiled = jax.jit(lambda v: SE3.hat(v, xp=jnp))
    compiled(jax.random.normal(jax.random.PRNGKey(0), (2, 6)))


def test_jit_so3_random():
    key = jax.random.PRNGKey(42)
    compiled = jax.jit(lambda k: SO3.random(2, key=k, xp=jnp))
    compiled(key)


def test_jit_se3_random():
    key = jax.random.PRNGKey(42)
    compiled = jax.jit(lambda k: SE3.random(2, key=k, xp=jnp))
    compiled(key)


def test_jit_se3_vee():
    compiled = jax.jit(lambda M: SE3.vee(M, xp=jnp))
    M = jnp.zeros((2, 4, 4)).at[:, 0, 1].set(-1.0).at[:, 1, 0].set(1.0).at[:, :3, 3].set(jax.random.normal(jax.random.PRNGKey(0), (2, 3)))
    compiled(M)
