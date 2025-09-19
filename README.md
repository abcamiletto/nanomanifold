# nanomanifold

Fast, batched and differentiable SO(3)/SE(3) transforms for any backend (NumPy, PyTorch, JAX, ...)

Works directly on arrays, defined as:

- **SO(3)**: unit quaternions `[w, x, y, z]` for 3D rotations, shape `(..., 4)`
- **SE(3)**: concatenated `[quat, translation]`, shape `(..., 7)`

```python
import numpy as np
from nanomanifold import SO3, SE3

# Rotations stored as quaternion arrays [w,x,y,z]
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
q3 = SO3.from_matrix(rotation_matrix)

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

# Compose transforms
T_combined = SE3.multiply(T1, T2)
T_inverse = SE3.inverse(T_combined)

# Apply to points
transformed = SE3.transform_points(T_combined, points)
```

## API Reference

All functions are available via `nanomanifold.SO3` and `nanomanifold.SE3`. Shapes follow the
Array API convention and accept arbitrarily batched inputs.

### SO3 (3D Rotations)

| Function                              | Signature                                 |
| ------------------------------------- | ----------------------------------------- |
| `canonicalize(q)`                     | `(...,4) -> (...,4)`                      |
| `to_axis_angle(q)`                    | `(...,4) -> (...,3)`                      |
| `from_axis_angle(axis_angle)`         | `(...,3) -> (...,4)`                      |
| `to_euler(q, convention="ZYX")`       | `(...,4) -> (...,3)`                      |
| `from_euler(euler, convention="ZYX")` | `(...,3) -> (...,4)`                      |
| `to_matrix(q)`                        | `(...,4) -> (...,3,3)`                    |
| `from_matrix(R)`                      | `(...,3,3) -> (...,4)`                    |
| `multiply(q1, q2)`                    | `(...,4), (...,4) -> (...,4)`             |
| `inverse(q)`                          | `(...,4) -> (...,4)`                      |
| `rotate_points(q, points)`            | `(...,4), (...,N,3) -> (...,N,3)`         |
| `slerp(q1, q2, t)`                    | `(...,4), (...,4), (...,N) -> (...,N,4)`  |
| `distance(q1, q2)`                    | `(...,4), (...,4) -> (...)`               |
| `log(q)`                              | `(...,4) -> (...,3)`                      |
| `exp(tangent)`                        | `(...,3) -> (...,4)`                      |
| `hat(w)`                              | `(...,3) -> (...,3,3)`                    |
| `vee(W)`                              | `(...,3,3) -> (...,3)`                    |
| `weighted_mean(quats, weights)`       | `sequence of (...,4), (...,N) -> (...,4)` |
| `mean(quats)`                         | `sequence of (...,4) -> (...,4)`          |

### SE3 (Rigid Transforms)

| Function                        | Signature                         |
| ------------------------------- | --------------------------------- |
| `canonicalize(se3)`             | `(...,7) -> (...,7)`              |
| `from_rt(quat, translation)`    | `(...,4), (...,3) -> (...,7)`     |
| `to_rt(se3)`                    | `(...,7) -> (quat, translation)`  |
| `from_matrix(T)`                | `(...,4,4) -> (...,7)`            |
| `to_matrix(se3)`                | `(...,7) -> (...,4,4)`            |
| `multiply(se3_1, se3_2)`        | `(...,7), (...,7) -> (...,7)`     |
| `inverse(se3)`                  | `(...,7) -> (...,7)`              |
| `transform_points(se3, points)` | `(...,7), (...,N,3) -> (...,N,3)` |
| `log(se3)`                      | `(...,7) -> (...,6)`              |
| `exp(tangent)`                  | `(...,6) -> (...,7)`              |
