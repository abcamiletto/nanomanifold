"""Lie-algebra maps of SE(3). Tangent vectors are ``(..., 6)`` arrays ``[ω, ρ]``: rotation first, then translation.

``exp`` and ``log`` apply the left Jacobian ``V = I + A·ω̂ + B·ω̂²`` and its inverse to the translation, with
coefficients that are functions of ``θ = |ω|``. Closed forms such as ``(θ - sin θ)/θ³`` lose about ``eps/θ²``
relative to cancellation, and their gradients more, so below a window the coefficients come from their series in
``θ²``, truncated after ``θ⁸``. The first dropped term is at most ``6.4e-9·θ¹⁰`` relative (the log coefficient's), so
the series is exact to machine precision up to ``θ = (eps / 6.4e-9)^(1/10)``, which is 0.18 rad in float64. The
window is capped at 1 rad: float32 gains nothing past it, and float16 would need the coefficients it underflows to
subnormals.
"""

from types import ModuleType
from typing import Any

from jaxtyping import Float

from nanomanifold import SO3, common
from nanomanifold.common import get_namespace
from nanomanifold.SO3.primitives.quaternion import QuaternionConvention

# Taylor coefficients in θ² of A = Σ (-θ²)ᵏ/(2k+2)! and B = Σ (-θ²)ᵏ/(2k+3)!.
_A = (1 / 2, -1 / 24, 1 / 720, -1 / 40320, 1 / 3628800)
_B = (1 / 6, -1 / 120, 1 / 5040, -1 / 362880, 1 / 39916800)
# Taylor coefficients in θ² of C = Σ |B₂ₖ₊₂|·θ²ᵏ/(2k+2)!, B the Bernoulli numbers.
_C = (1 / 12, 1 / 720, 1 / 30240, 1 / 1209600, 1 / 47900160)


def exp(xi: Float[Any, "... 6"], *, convention: QuaternionConvention = "wxyz", xp: ModuleType | None = None) -> Float[Any, "... 7"]:
    """Map se(3) tangent vectors ``[ω, ρ]`` ``(..., 6)`` to transforms ``(..., 7)``: ``[SO3.exp(ω), V ρ]``."""
    if xp is None:
        xp = get_namespace(xi)

    omega = xi[..., :3]
    rho = xi[..., 3:6]

    # V = I + A·ω̂ + B·ω̂², with A = (1 - cos θ)/θ² written as 2·sin²(θ/2)/θ², free of cancellation.
    theta_sq, small, theta = _small_angle(omega, xp)
    A = xp.where(small, _series(theta_sq, _A), 2.0 * xp.sin(theta / 2.0) ** 2 / theta**2)
    B = xp.where(small, _series(theta_sq, _B), (theta - xp.sin(theta)) / theta**3)
    t = _apply(omega, A, B, rho, xp)

    return xp.concatenate([SO3.exp(omega, convention=convention, xp=xp), t], axis=-1)


def log(T: Float[Any, "... 7"], *, convention: QuaternionConvention = "wxyz", xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    """Map transforms ``(..., 7)`` to se(3) tangent vectors ``[ω, ρ]`` ``(..., 6)``: ``[SO3.log(q), V⁻¹ t]``."""
    if xp is None:
        xp = get_namespace(T)

    omega = SO3.log(T[..., :4], convention=convention, xp=xp)
    t = T[..., 4:7]

    # V⁻¹ = I - ω̂/2 + C·ω̂², with C = (1 - (θ/2)·cot(θ/2))/θ²; θ ≤ π here, so sin(θ/2) > 0.
    theta_sq, small, theta = _small_angle(omega, xp)
    cot_half = xp.cos(theta / 2.0) / xp.sin(theta / 2.0)
    C = xp.where(small, _series(theta_sq, _C), (1.0 - 0.5 * theta * cot_half) / theta**2)
    rho = _apply(omega, -0.5, C, t, xp)

    return xp.concatenate([omega, rho], axis=-1)


def hat(xi: Float[Any, "... 6"], *, xp: ModuleType | None = None) -> Float[Any, "... 4 4"]:
    """4x4 matrices ``[[SO3.hat(ω), ρ], [0, 0]]`` ``(..., 4, 4)`` of tangent vectors ``[ω, ρ]`` ``(..., 6)``."""
    if xp is None:
        xp = get_namespace(xi)
    top = xp.concatenate([SO3.hat(xi[..., :3], xp=xp), xi[..., 3:6, None]], axis=-1)  # (..., 3, 4)
    return xp.concatenate([top, xp.zeros_like(top[..., :1, :])], axis=-2)


def vee(Xi: Float[Any, "... 4 4"], *, xp: ModuleType | None = None) -> Float[Any, "... 6"]:
    """Inverse of ``hat``: the tangent vectors ``[ω, ρ]`` ``(..., 6)`` of se(3) matrices ``(..., 4, 4)``."""
    if xp is None:
        xp = get_namespace(Xi)
    return xp.concatenate([SO3.vee(Xi[..., :3, :3], xp=xp), Xi[..., :3, 3]], axis=-1)


def _small_angle(omega: Any, xp: ModuleType) -> tuple[Any, Any, Any]:
    """Return ``θ²``, a mask of the angles inside the series window and ``θ`` set to 1 inside it, each ``(..., 1)``.

    Closed forms divided by the returned ``θ`` stay finite in the masked branch, gradients included.
    """
    eps = common.safe_eps(omega.dtype, xp, scale=1.0)
    window = min((eps / 6.4e-9) ** 0.1, 1.0)
    theta_sq = xp.sum(omega * omega, axis=-1, keepdims=True)
    small = theta_sq < window**2
    theta = xp.sqrt(xp.where(small, xp.ones_like(theta_sq), theta_sq))
    return theta_sq, small, theta


def _series(theta_sq: Any, coefficients: tuple[float, ...]) -> Any:
    """Evaluate ``Σₖ coefficients[k]·θ^(2k)`` by Horner's rule."""
    result = coefficients[-1]
    for c in coefficients[-2::-1]:
        result = c + theta_sq * result
    return result


def _apply(omega: Any, a: Any, b: Any, v: Any, xp: ModuleType) -> Any:
    """Return ``(I + a·ω̂ + b·ω̂²) v = v + ω × (a·v + b·(ω × v))`` for ``a``, ``b`` of shape ``(..., 1)``.

    Two cross products instead of the 3x3 matrix: no ``hat``, identity or batched ``matmul``. Adding ``v`` last keeps
    small angles more accurate than the matrix; near π the matrix is ~0.3 ulp better at p99, in every dtype.
    """
    return v + common.cross(omega, a * v + b * common.cross(omega, v, xp), xp)
