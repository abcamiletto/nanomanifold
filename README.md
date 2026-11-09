# nanomanifold

Batched, differentiable SO(3) and SE(3) operations on plain arrays, for NumPy, PyTorch and JAX.

- **Any backend.** Written once against the [Array API](https://data-apis.org/array-api/); the backend is
  picked from the input array.
- **Plain arrays.** No wrapper classes: rotations are `(..., 4)` quaternions, rigid transforms are `(..., 7)`
  `[quaternion, translation]` arrays.
- **Batched.** Every function accepts arbitrary leading batch dimensions.
- **Differentiable and compilable.** Numerically careful around the identity, half turns and gimbal lock, and
  every function traces under `torch.compile(fullgraph=True)` and `jax.jit`.

```bash
pip install nanomanifold
```

## Quick start

```python
import numpy as np
from nanomanifold import SE3, SO3

q1 = SO3.from_axis_angle(np.array([np.pi / 2, 0.0, 0.0]))  # 90° about x
q2 = SO3.from_euler(np.array([np.pi / 4, 0.0, 0.0]))      # 45° about z ("ZYX")
q = SO3.multiply(q1, q2)                                   # q2 first, then q1

points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
rotated = SO3.rotate_points(q, points)
halfway = SO3.slerp(q1, q2, np.array([0.5]))               # (1, 4)
angle = SO3.distance(q1, q2)                               # geodesic distance, radians

T = SE3.from_rt(q, np.array([1.0, 2.0, 3.0]))
moved = SE3.transform_points(T, points)
matrix = SE3.to_matrix(SE3.inverse(T))                     # (4, 4)
```

Batch dimensions broadcast through everything:

```python
q = SO3.random(32, 10)                                     # (32, 10, 4)
SO3.rotate_points(q, np.zeros((32, 10, 100, 3)))           # (32, 10, 100, 3)
```

## Conventions

- Quaternions are `wxyz` by default. Pass `convention="xyzw"` to any function to read and write `xyzw`.
- Returned quaternions are canonical: unit norm with `w >= 0`.
- `multiply(a, b)` composes like rotation matrices, `R(a) @ R(b)`, so `b` is applied first.
- Euler conventions are three-letter strings: uppercase is intrinsic (`"ZYX"`, the default), lowercase is
  extrinsic (`"xyz"`). Proper Euler sequences such as `"ZXZ"` are supported.

## Representations

| Name         | Shape         | Options                       | Notes                                           |
| ------------ | ------------- | ----------------------------- | ----------------------------------------------- |
| `quat`       | `(..., 4)`    | `convention="wxyz" \| "xyzw"` | Unit quaternion                                 |
| `axis_angle` | `(..., 3)`    |                               | Rotation vector: axis scaled by the angle       |
| `hinge`      | `(..., 1)`    | `axes` (required)             | Coefficient along a fixed axis: `angles * axes` |
| `rotmat`     | `(..., 3, 3)` |                               | Rotation matrix                                 |
| `matrix`     | `(..., 3, 3)` | `mode="svd" \| "davenport"`   | Any 3x3 matrix, projected onto SO(3) when read  |
| `sixd`       | `(..., 6)`    |                               | First two rotation-matrix columns (Zhou et al.) |
| `euler`      | `(..., 3)`    | `convention="ZYX"`, ...       | Euler angles                                    |

`SO3.convert` converts between any two of them. Options go in `src_kwargs` and `dst_kwargs`:

```python
R = SO3.convert(sixd, src="sixd", dst="rotmat")
q = SO3.convert(R, src="rotmat", dst="quat", dst_kwargs={"convention": "xyzw"})
e = SO3.convert(q, src="quat", dst="euler", src_kwargs={"convention": "xyzw"}, dst_kwargs={"convention": "XYZ"})
R = SO3.convert(noisy, src="matrix", dst="rotmat", src_kwargs={"mode": "davenport"})
angles = SO3.convert(R, src="rotmat", dst="hinge", dst_kwargs={"axes": axes})
```

The representations form a tree, and each one only defines the step to its parent and back:

```
canonical quaternion
├── quat
├── axis_angle
│   └── hinge
└── rotmat
    ├── sixd
    ├── euler
    └── matrix
```

A conversion goes up from `src` to the closest common ancestor and back down to `dst`. Each pair has exactly
one route, and it only uses native steps: `sixd -> rotmat` is a single Gram-Schmidt step and never passes
through a quaternion. Converting a representation to itself goes to its parent and back, which normalizes it
and applies any change of options (for example `euler` `"XYZ"` to `"ZYX"`).

`SO3.from_<rep>` and `SO3.to_<rep>` are shorthands for converting from and to `quat`.

## API

All functions live in `nanomanifold.SO3` and `nanomanifold.SE3` and accept `convention=` for the quaternion
order and `xp=` for the backend.

### SO3

| Function                                                           | Shapes                                      |
| ------------------------------------------------------------------ | ------------------------------------------- |
| `convert(x, src, dst, src_kwargs={}, dst_kwargs={})`               | any representation -> any representation    |
| `from_axis_angle(axis_angle)` / `to_axis_angle(q)`                 | `(...,3) <-> (...,4)`                       |
| `from_hinge(angles, axes)` / `to_hinge(q, axes)`                   | `(...,1) <-> (...,4)`                       |
| `from_euler(e, convention="ZYX")` / `to_euler(q, convention="ZYX")` | `(...,3) <-> (...,4)`                    |
| `from_rotmat(R)` / `to_rotmat(q)`                                  | `(...,3,3) <-> (...,4)`                     |
| `from_matrix(M, mode="svd")`                                       | `(...,3,3) -> (...,4)`                      |
| `from_sixd(sixd)` / `to_sixd(q)`                                   | `(...,6) <-> (...,4)`                       |
| `from_quat(q, convention)` / `to_quat(q, convention)`              | `(...,4) <-> (...,4)`, reorders to/from `wxyz` |
| `canonicalize(q)`                                                  | `(...,4) -> (...,4)`                        |
| `identity_as(ref, batch_dims, rotation_type="quat")`               | identity in any representation              |
| `random(*shape)`                                                   | `-> (*shape, 4)`, uniform on SO(3)          |
| `multiply(q1, q2)`                                                 | `(...,4), (...,4) -> (...,4)`               |
| `inverse(q)`                                                       | `(...,4) -> (...,4)`                        |
| `rotate_points(q, points)`                                         | `(...,4), (...,N,3) -> (...,N,3)`           |
| `slerp(q1, q2, t)`                                                 | `(...,4), (...,4), (...,N) -> (...,N,4)`    |
| `distance(r1, r2, rotation_type="quat", rot_kwargs={})`            | `-> (...)`, angle in `[0, π]`               |
| `log(q)` / `exp(tangent)`                                          | `(...,4) <-> (...,3)`                       |
| `hat(w)` / `vee(W)`                                                | `(...,3) <-> (...,3,3)`                     |
| `weighted_mean(quats, weights)` / `mean(quats)`                    | sequence of `(...,4)` -> `(...,4)`          |

### SE3

| Function                                                           | Shapes                                      |
| ------------------------------------------------------------------ | ------------------------------------------- |
| `from_rt(q, t)` / `to_rt(T)`                                       | `(...,4), (...,3) <-> (...,7)`              |
| `from_matrix(M)` / `to_matrix(T)`                                  | `(...,4,4) <-> (...,7)`                     |
| `canonicalize(T)`                                                  | `(...,7) -> (...,7)`                        |
| `random(*shape)`                                                   | `-> (*shape, 7)`                            |
| `multiply(T1, T2)`                                                 | `(...,7), (...,7) -> (...,7)`               |
| `inverse(T)`                                                       | `(...,7) -> (...,7)`                        |
| `transform_points(T, points)`                                      | `(...,7), (...,N,3) -> (...,N,3)`           |
| `slerp(T1, T2, t)`                                                 | `(...,7), (...,7), (...,N) -> (...,N,7)`    |
| `log(T)` / `exp(tangent)`                                          | `(...,7) <-> (...,6)`                       |
| `hat(v)` / `vee(M)`                                                | `(...,6) <-> (...,4,4)`                     |
| `weighted_mean(Ts, weights)` / `mean(Ts)`                          | sequence of `(...,7)` -> `(...,7)`          |

## Backends and compilation

The backend is inferred from the input array, so the same code runs on NumPy, PyTorch and JAX, and output
stays on the input's device and dtype. Functions that take no input array (`random`) default to NumPy; pass
`xp=` to choose another backend, and `key=` for JAX.

```python
import jax
import jax.numpy as jnp
import torch

q = SO3.random(8, xp=torch)
q = SO3.random(8, key=jax.random.PRNGKey(0), xp=jnp)


@torch.compile(fullgraph=True)
def step(q, points):
    return SO3.rotate_points(q, points), SO3.log(q)


log_q = jax.jit(SO3.log)(q)
```

Passing `xp=` explicitly (for example `xp=torch`) skips backend inference, but compiling works without it.
