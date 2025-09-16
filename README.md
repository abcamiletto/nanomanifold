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

### SO3 (3D Rotations)

| Function                       | Input → Output                      | Description                 |
| ------------------------------ | ----------------------------------- | --------------------------- |
| `from_axis_angle(axis, angle)` | `(...,3), (...) → (...,4)`          | Create from axis-angle      |
| `from_euler(angles)`           | `(...,3) → (...,4)`                 | Create from Euler angles    |
| `from_matrix(R)`               | `(...,3,3) → (...,4)`               | Create from rotation matrix |
| `multiply(q1, q2)`             | `(...,4), (...,4) → (...,4)`        | Compose rotations           |
| `slerp(q1, q2, t)`             | `(...,4), (...,4), (...) → (...,4)` | Spherical interpolation     |
| `rotate_points(q, points)`     | `(...,4), (...,N,3) → (...,N,3)`    | Rotate 3D points            |

### SE3 (Rigid Transforms)

| Function                      | Input → Output                   | Description                        |
| ----------------------------- | -------------------------------- | ---------------------------------- |
| `from_rt(q, t)`               | `(...,4), (...,3) → (...,7)`     | Create from rotation + translation |
| `from_matrix(T)`              | `(...,4,4) → (...,7)`            | Create from 4×4 matrix             |
| `multiply(T1, T2)`            | `(...,7), (...,7) → (...,7)`     | Compose transforms                 |
| `inverse(T)`                  | `(...,7) → (...,7)`              | Invert transform                   |
| `transform_points(T, points)` | `(...,7), (...,N,3) → (...,N,3)` | Transform 3D points                |
