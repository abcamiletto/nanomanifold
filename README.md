# nanomanifold

Fast, batched and differentiable SO3/SE3 transforms for any backend (NumPy, PyTorch, JAX, ...)

Works directly on arrays, defined as:

- **SO3**: unit quaternions `[w, x, y, z]` for 3D rotations, shape `(..., 4)`
- **SE3**: concatenated `[quat, translation]`, shape `(..., 7)`

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

| Function                              | Signature                                 | Description                                                   |
| ------------------------------------- | ----------------------------------------- | ------------------------------------------------------------- |
| `canonicalize(q)`                     | `(...,4) -> (...,4)`                      | Normalize quaternions and enforce a non-negative scalar part. |
| `to_axis_angle(q)`                    | `(...,4) -> (...,3)`                      | Convert quaternion to axis-angle tangent vector.              |
| `from_axis_angle(axis_angle)`         | `(...,3) -> (...,4)`                      | Build quaternion from axis-angle tangent vector.              |
| `to_euler(q, convention="ZYX")`       | `(...,4) -> (...,3)`                      | Convert quaternion to intrinsic or extrinsic Euler angles.    |
| `from_euler(euler, convention="ZYX")` | `(...,3) -> (...,4)`                      | Create quaternion matching the requested Euler convention.    |
| `to_matrix(q)`                        | `(...,4) -> (...,3,3)`                    | Convert quaternion to a 3×3 rotation matrix.                  |
| `from_matrix(R)`                      | `(...,3,3) -> (...,4)`                    | Build quaternion from a rotation matrix.                      |
| `multiply(q1, q2)`                    | `(...,4), (...,4) -> (...,4)`             | Compose two rotations (`q2` then `q1`).                       |
| `inverse(q)`                          | `(...,4) -> (...,4)`                      | Return the quaternion inverse.                                |
| `rotate_points(q, points)`            | `(...,4), (...,N,3) -> (...,N,3)`         | Rotate 3D point sets by `q`.                                  |
| `slerp(q1, q2, t)`                    | `(...,4), (...,4), (...,N) -> (...,N,4)`  | Short-arc spherical interpolation for t ∈ [0, 1].             |
| `distance(q1, q2)`                    | `(...,4), (...,4) -> (...)`               | Geodesic angular distance between rotations.                  |
| `log(q)`                              | `(...,4) -> (...,3)`                      | Logarithmic map from SO(3) to the tangent space.              |
| `exp(tangent)`                        | `(...,3) -> (...,4)`                      | Exponential map from so(3) tangent vectors to SO(3).          |
| `hat(w)`                              | `(...,3) -> (...,3,3)`                    | Convert ω vectors to skew-symmetric matrices.                 |
| `vee(W)`                              | `(...,3,3) -> (...,3)`                    | Recover ω vectors from skew-symmetric matrices.               |
| `weighted_mean(quats, weights)`       | `sequence of (...,4), (...,N) -> (...,4)` | Weighted Markley quaternion mean with canonicalization.       |
| `mean(quats)`                         | `sequence of (...,4) -> (...,4)`          | Uniformly weighted quaternion mean.                           |

### SE3 (Rigid Transforms)

| Function                        | Signature                         | Description                                               |
| ------------------------------- | --------------------------------- | --------------------------------------------------------- |
| `canonicalize(se3)`             | `(...,7) -> (...,7)`              | Normalize the rotation part and keep translations intact. |
| `from_rt(quat, translation)`    | `(...,4), (...,3) -> (...,7)`     | Construct SE(3) tuples `[quat, t]`.                       |
| `to_rt(se3)`                    | `(...,7) -> (quat, translation)`  | Split SE(3) tuples into rotation and translation.         |
| `from_matrix(T)`                | `(...,4,4) -> (...,7)`            | Convert homogeneous matrices to SE(3) tuples.             |
| `to_matrix(se3)`                | `(...,7) -> (...,4,4)`            | Convert SE(3) tuples to 4×4 homogeneous matrices.         |
| `multiply(se3_1, se3_2)`        | `(...,7), (...,7) -> (...,7)`     | Compose transforms (`se3_2` then `se3_1`).                |
| `inverse(se3)`                  | `(...,7) -> (...,7)`              | Invert rigid transforms.                                  |
| `transform_points(se3, points)` | `(...,7), (...,N,3) -> (...,N,3)` | Apply rotation and translation to point clouds.           |
| `log(se3)`                      | `(...,7) -> (...,6)`              | Logarithmic map returning `[ω, ρ]` in se(3).              |
| `exp(tangent)`                  | `(...,6) -> (...,7)`              | Exponential map from se(3) tangent vectors to SE(3).      |
