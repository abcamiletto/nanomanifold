"""Pairwise SO(3) representation conversions.

Every function converts directly from one rotation representation to another
without requiring the caller to chain two calls through the internal
quaternion format.

Representations
---------------
- ``axis_angle``  -- axis-angle vector, shape ``(..., 3)``
- ``euler``       -- Euler angles, shape ``(..., 3)``
- ``matrix``      -- rotation matrix, shape ``(..., 3, 3)``
- ``quat_wxyz``   -- unit quaternion ``[w, x, y, z]``, shape ``(..., 4)``
- ``quat_xyzw``   -- unit quaternion ``[x, y, z, w]``, shape ``(..., 4)``
- ``sixd``        -- 6-D rotation (first two columns of matrix), shape ``(..., 6)``

Usage::

    from nanomanifold import SO3

    matrix = SO3.conversions.from_axis_angle_to_matrix(axis_angle)

To add a specialised (direct) conversion, create a new module inside this
package and import it here, overriding the generic version from ``_generic``.
"""

# Default: all conversions go through quaternion
from ._generic import (
    from_axis_angle_to_euler,
    from_axis_angle_to_matrix,
    from_axis_angle_to_quat_wxyz,
    from_axis_angle_to_quat_xyzw,
    from_axis_angle_to_sixd,
    from_euler_to_axis_angle,
    from_euler_to_matrix,
    from_euler_to_quat_wxyz,
    from_euler_to_quat_xyzw,
    from_euler_to_sixd,
    from_matrix_to_axis_angle,
    from_matrix_to_euler,
    from_matrix_to_quat_wxyz,
    from_matrix_to_quat_xyzw,
    from_matrix_to_sixd,
    from_quat_wxyz_to_axis_angle,
    from_quat_wxyz_to_euler,
    from_quat_wxyz_to_matrix,
    from_quat_wxyz_to_quat_xyzw,
    from_quat_wxyz_to_sixd,
    from_quat_xyzw_to_axis_angle,
    from_quat_xyzw_to_euler,
    from_quat_xyzw_to_matrix,
    from_quat_xyzw_to_quat_wxyz,
    from_quat_xyzw_to_sixd,
    from_sixd_to_axis_angle,
    from_sixd_to_euler,
    from_sixd_to_matrix,
    from_sixd_to_quat_wxyz,
    from_sixd_to_quat_xyzw,
)

# To override a generic conversion with a specialised one, add an import here:
# from ._axis_angle_to_matrix import from_axis_angle_to_matrix  # noqa: F811
from ._euler_to_matrix import from_euler_to_matrix  # noqa: F811
from ._matrix_to_euler import from_matrix_to_euler  # noqa: F811
from ._matrix_to_sixd import from_matrix_to_sixd  # noqa: F811
from ._sixd_to_matrix import from_sixd_to_matrix  # noqa: F811

__all__ = [
    "from_axis_angle_to_euler",
    "from_axis_angle_to_matrix",
    "from_axis_angle_to_quat_wxyz",
    "from_axis_angle_to_quat_xyzw",
    "from_axis_angle_to_sixd",
    "from_euler_to_axis_angle",
    "from_euler_to_matrix",
    "from_euler_to_quat_wxyz",
    "from_euler_to_quat_xyzw",
    "from_euler_to_sixd",
    "from_matrix_to_axis_angle",
    "from_matrix_to_euler",
    "from_matrix_to_quat_wxyz",
    "from_matrix_to_quat_xyzw",
    "from_matrix_to_sixd",
    "from_quat_wxyz_to_axis_angle",
    "from_quat_wxyz_to_euler",
    "from_quat_wxyz_to_matrix",
    "from_quat_wxyz_to_quat_xyzw",
    "from_quat_wxyz_to_sixd",
    "from_quat_xyzw_to_axis_angle",
    "from_quat_xyzw_to_euler",
    "from_quat_xyzw_to_matrix",
    "from_quat_xyzw_to_quat_wxyz",
    "from_quat_xyzw_to_sixd",
    "from_sixd_to_axis_angle",
    "from_sixd_to_euler",
    "from_sixd_to_matrix",
    "from_sixd_to_quat_wxyz",
    "from_sixd_to_quat_xyzw",
]
