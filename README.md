# nanomanifold

Fast, batched and differentiable SO(3)/SE(3) transforms for any backend (NumPy, PyTorch, JAX, ...)

Works directly on arrays, defined as:

- **SO(3)**: unit `quat` quaternions with explicit `convention="wxyz"` or `convention="xyzw"`, shape `(..., 4)`
- **SE(3)**: concatenated `[quat, translation]`, shape `(..., 7)`

```python
import numpy as np
from nanomanifold import SO3, SE3

# Rotations stored as quaternion arrays, default convention="wxyz"
q = SO3.from_axis_angle(np.array([0, 0, 1]), np.pi/4)  # 45° around Z
points = np.array([[1, 0, 0], [0, 1, 0]])
rotated = SO3.rotate_points(q, points)

# Rigid transforms stored as 7D arrays [quat, translation]
T = SE3.from_rt(q, np.array([1, 0, 0]))  # rotation + translation
transformed = SE3.transform_points(T, points)
```

## Installation

```bash
pip install nanomanifold
```

## Quick Start

### Rotations (SO3)

```python
from nanomanifold import SO3

# Create rotations
q1 = SO3.from_axis_angle([1, 0, 0], np.pi/2)    # 90° around X
q2 = SO3.from_euler([0, 0, np.pi/4])            # 45° around Z
q3 = SO3.from_rotmat(rotation_matrix)

# Compose and interpolate
q_combined = SO3.multiply(q1, q2)
q_halfway = SO3.slerp(q1, q2, t=0.5)

# Apply to points
points = np.array([[1, 0, 0], [0, 1, 0]])
rotated = SO3.rotate_points(q_combined, points)
```

### Rigid Transforms (SE3)

```python
from nanomanifold import SE3

# Create transforms
T1 = SE3.from_rt(q1, [1, 2, 3])               # rotation + translation
T2 = SE3.from_matrix(transformation_matrix)

# Compose and interpolate
T_combined = SE3.multiply(T1, T2)
T_inverse = SE3.inverse(T_combined)
T_halfway = SE3.slerp(T1, T2, t=0.5)

# Apply to points
transformed = SE3.transform_points(T_combined, points)
```

## API Reference

All functions are available via `nanomanifold.SO3` and `nanomanifold.SE3`. Shapes follow the
Array API convention and accept arbitrarily batched inputs.

### SO3 (3D Rotations)

Supported SO3 parametrizations:

| Name | Shape | Notes |
| ---- | ----- | ----- |
| `quat` | `(...,4)` | Unit quaternion with explicit `convention="wxyz"` or `convention="xyzw"` |
| `axis_angle` | `(...,3)` | Rotation vector / axis-angle parametrization |
| `euler` | `(...,3)` | Euler angles with an explicit convention such as `"ZYX"` |
| `rotmat` | `(...,3,3)` | Normalized rotation matrix in SO(3) |
| `matrix` | `(...,3,3)` | Generic 9D matrix, not assumed to be normalized |
| `sixd` | `(...,6)` | 6D continuous representation built from the first two rotation-matrix columns |

| Function                              | Signature                                 |
| ------------------------------------- | ----------------------------------------- |
| `canonicalize(q, convention="wxyz")`  | `(...,4) -> (...,4)`                      |
| `to_axis_angle(q, convention="wxyz")` | `(...,4) -> (...,3)`                      |
| `from_axis_angle(axis_angle, convention="wxyz")` | `(...,3) -> (...,4)`         |
| `to_euler(q, convention="ZYX", quat_convention="wxyz")` | `(...,4) -> (...,3)`     |
| `from_euler(euler, convention="ZYX", quat_convention="wxyz")` | `(...,3) -> (...,4)` |
| `convert(x, src=..., dst=...)`        | dynamic                                   |
| `identity_as(ref, batch_dims=..., rotation_type=..., convention="wxyz")` | dynamic |
| `to_rotmat(q, convention="wxyz")`     | `(...,4) -> (...,3,3)`                    |
| `from_rotmat(R, convention="wxyz")`   | `(...,3,3) -> (...,4)`                    |
| `from_matrix(R, convention="wxyz", mode="svd")` | `(...,3,3) -> (...,4)`           |
| `from_quat(quat, convention="xyzw")`  | `(...,4) -> (...,4)`                      |
| `to_quat(quat, convention="xyzw")`    | `(...,4) -> (...,4)`                      |
| `to_sixd(q, convention="wxyz")`       | `(...,4) -> (...,6)`                      |
| `from_sixd(sixd, convention="wxyz")`  | `(...,6) -> (...,4)`                      |
| `multiply(q1, q2, convention="wxyz")` | `(...,4), (...,4) -> (...,4)`             |
| `inverse(q, convention="wxyz")`       | `(...,4) -> (...,4)`                      |
| `rotate_points(q, points, convention="wxyz")` | `(...,4), (...,N,3) -> (...,N,3)` |
| `slerp(q1, q2, t, convention="wxyz")` | `(...,4), (...,4), (...,N) -> (...,N,4)`  |
| `distance(q1, q2, rotation_type="quat", convention="wxyz")` | dynamic                  |
| `log(q, convention="wxyz")`           | `(...,4) -> (...,3)`                      |
| `exp(tangent, convention="wxyz")`     | `(...,3) -> (...,4)`                      |
| `hat(w)`                              | `(...,3) -> (...,3,3)`                    |
| `vee(W)`                              | `(...,3,3) -> (...,3)`                    |
| `weighted_mean(quats, weights, convention="wxyz")` | `sequence of (...,4), (...,N) -> (...,4)` |
| `mean(quats, convention="wxyz")`      | `sequence of (...,4) -> (...,4)`          |
| `random(*shape, convention="wxyz")`   | `(...,4)`                                 |

### SE3 (Rigid Transforms)

| Function                              | Signature                                 |
| ------------------------------------- | ----------------------------------------- |
| `canonicalize(se3, convention="wxyz")` | `(...,7) -> (...,7)`                     |
| `from_rt(quat, translation, convention="wxyz")` | `(...,4), (...,3) -> (...,7)`   |
| `to_rt(se3, convention="wxyz")`       | `(...,7) -> (quat, translation)`          |
| `from_matrix(T)`                        | `(...,4,4) -> (...,7)`             |
| `to_matrix(se3, convention="wxyz")`   | `(...,7) -> (...,4,4)`                    |
| `multiply(se3_1, se3_2, convention="wxyz")` | `(...,7), (...,7) -> (...,7)`      |
| `inverse(se3)`                        | `(...,7) -> (...,7)`                      |
| `transform_points(se3, points)`       | `(...,7), (...,N,3) -> (...,N,3)`         |
| `slerp(se3_1, se3_2, t)`             | `(...,7), (...,7), (...,N) -> (...,N,7)`  |
| `log(se3)`                            | `(...,7) -> (...,6)`                      |
| `exp(tangent)`                        | `(...,6) -> (...,7)`                      |
| `hat(v)`                              | `(...,6) -> (...,4,4)`                    |
| `vee(M)`                              | `(...,4,4) -> (...,6)`                    |
| `weighted_mean(transforms, weights)`  | `sequence of (...,7), (...,N) -> (...,7)` |
| `mean(transforms)`                    | `sequence of (...,7) -> (...,7)`          |
| `random(*shape)`                      | `(...,7)`                                 |

## Pairwise Conversions (`SO3.conversions`)

Convert directly between any two rotation representations without going through
quaternions manually. All 30 pairwise functions follow the naming pattern
`from_{source}_to_{target}`.

Representations: `axis_angle`, `euler`, `matrix`, `rotmat`, `quat`, `sixd`.
Quaternion conventions use lowercase `convention="wxyz"` or `convention="xyzw"`.

| Function                                                        | Signature                   |
| --------------------------------------------------------------- | --------------------------- |
| `SO3.conversions.from_axis_angle_to_rotmat(aa)`                | `(...,3) -> (...,3,3)`      |
| `SO3.conversions.from_axis_angle_to_euler(aa, convention)`      | `(...,3) -> (...,3)`        |
| `SO3.conversions.from_axis_angle_to_quat(aa, convention="wxyz")` | `(...,3) -> (...,4)`      |
| `SO3.conversions.from_axis_angle_to_sixd(aa)`                   | `(...,3) -> (...,6)`        |
| `SO3.conversions.from_euler_to_axis_angle(e, convention)`       | `(...,3) -> (...,3)`        |
| `SO3.conversions.from_euler_to_rotmat(e, convention)`          | `(...,3) -> (...,3,3)`      |
| `SO3.conversions.from_euler_to_quat(e, src_convention="ZYX", dst_convention="wxyz")` | `(...,3) -> (...,4)` |
| `SO3.conversions.from_euler_to_sixd(e, convention)`             | `(...,3) -> (...,6)`        |
| `SO3.conversions.from_rotmat_to_axis_angle(R)`                 | `(...,3,3) -> (...,3)`      |
| `SO3.conversions.from_rotmat_to_euler(R, convention)`          | `(...,3,3) -> (...,3)`      |
| `SO3.conversions.from_rotmat_to_quat(R, convention="wxyz")`   | `(...,3,3) -> (...,4)`      |
| `SO3.conversions.from_rotmat_to_sixd(R)`                      | `(...,3,3) -> (...,6)`      |
| `SO3.conversions.from_matrix_to_rotmat(M, mode="svd")`       | `(...,3,3) -> (...,3,3)`    |
| `SO3.conversions.from_matrix_to_axis_angle(M, mode="svd")` | `(...,3,3) -> (...,3)` |
| `SO3.conversions.from_matrix_to_euler(R, convention, mode="svd")` | `(...,3,3) -> (...,3)` |
| `SO3.conversions.from_matrix_to_quat(R, convention="wxyz", mode="svd")` | `(...,3,3) -> (...,4)` |
| `SO3.conversions.from_matrix_to_sixd(R, mode="svd")`       | `(...,3,3) -> (...,6)` |
| `SO3.conversions.from_quat_to_axis_angle(q, convention="wxyz")` | `(...,4) -> (...,3)`       |
| `SO3.conversions.from_quat_to_euler(q, src_convention="wxyz", dst_convention="ZYX")` | `(...,4) -> (...,3)` |
| `SO3.conversions.from_quat_to_rotmat(q, convention="wxyz")`    | `(...,4) -> (...,3,3)`      |
| `SO3.conversions.from_quat_to_quat(q, src_convention="wxyz", dst_convention="xyzw")` | `(...,4) -> (...,4)` |
| `SO3.conversions.from_quat_to_sixd(q, convention="wxyz")`      | `(...,4) -> (...,6)`        |
| `SO3.conversions.from_sixd_to_axis_angle(sixd)`                 | `(...,6) -> (...,3)`        |
| `SO3.conversions.from_sixd_to_euler(sixd, convention)`          | `(...,6) -> (...,3)`        |
| `SO3.conversions.from_sixd_to_rotmat(sixd)`                    | `(...,6) -> (...,3,3)`      |
| `SO3.conversions.from_sixd_to_quat(sixd, convention="wxyz")`    | `(...,6) -> (...,4)`        |

For runtime-selected conversions, use `SO3.convert`. `src="matrix"` treats the
input as a generic `3x3` matrix and projects it to `rotmat` before converting.
Euler uses the usual axis-order convention strings:

```python
matrix = SO3.convert(axis_angle, src="axis_angle", dst="matrix")
rotmat = SO3.convert(matrix, src="matrix", dst="rotmat")
quat_alt = SO3.convert(euler, src="euler", dst="quat", src_convention="XYZ", dst_convention="xyzw")
quat = SO3.convert(quat_alt, src="quat", dst="quat", src_convention="xyzw", dst_convention="wxyz")
euler = SO3.convert(rotmat, src="rotmat", dst="euler", dst_convention="ZYX")
```

## Backend-Explicit Mode

By default, nanomanifold auto-detects the array backend via `array_api_compat`. Every function also
accepts an optional `xp` keyword argument to specify the backend explicitly. This is required for
`torch.compile(fullgraph=True)`, since Dynamo cannot trace the dynamic dispatch:

```python
import torch
from nanomanifold import SO3, SE3

@torch.compile(fullgraph=True)
def forward(q1, q2, T1, T2):
    q_mid = SO3.slerp(q1, q2, torch.tensor([0.5]), xp=torch)
    T_mid = SE3.slerp(T1, T2, torch.tensor([0.5]), xp=torch)
    return q_mid, T_mid
```
