"""Throwaway benchmark for SO(3) projection via SVD vs Newton-Schulz.

Examples:
    uv run python scratch/compare_so3_projection.py --backend numpy
    uv run python scratch/compare_so3_projection.py --backend torch --device cpu
    uv run python scratch/compare_so3_projection.py --backend jax --jit
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from nanomanifold import SO3
from nanomanifold.SO3.primitives.rotmat import _project_matrix_to_rotmat as project_to_so3_svd

Array = Any


@dataclass
class Backend:
    name: str
    xp: Any
    device: str | None = None
    jax: Any | None = None

    def asarray(self, array: np.ndarray) -> Array:
        if self.name == "torch":
            return self.xp.tensor(array, device=self.device)
        return self.xp.asarray(array)

    def to_numpy(self, array: Array) -> np.ndarray:
        if self.name == "torch":
            return array.detach().cpu().numpy()
        return np.asarray(array)

    def sync(self, array: Array) -> None:
        if self.name == "torch":
            if array.device.type == "cuda":
                self.xp.cuda.synchronize(array.device)
            elif array.device.type == "mps":
                self.xp.mps.synchronize()
            return
        if self.name == "jax":
            array.block_until_ready()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["numpy", "torch", "jax"], default="numpy")
    parser.add_argument("--device", default=None, help="Torch device such as cpu, cuda, or mps.")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5, help="Newton-Schulz iteration count.")
    parser.add_argument(
        "--refinement-steps",
        type=int,
        default=3,
        help="Extra full-precision orthogonalization steps for the adapted torch variant.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[1e-3, 1e-2, 5e-2],
        help="Additive Gaussian noise applied to each rotation matrix.",
    )
    parser.add_argument(
        "--jit",
        action="store_true",
        help="JIT the JAX benchmark functions. Ignored for other backends.",
    )
    return parser.parse_args()


def load_backend(name: str, device: str | None) -> Backend:
    if name == "numpy":
        return Backend(name="numpy", xp=np)
    if name == "torch":
        import torch

        return Backend(name="torch", xp=torch, device=device or "cpu")
    if name == "jax":
        import jax
        import jax.numpy as jnp

        return Backend(name="jax", xp=jnp, jax=jax)
    raise ValueError(name)


def random_quaternions(rng: np.random.Generator, batch_size: int, dtype: np.dtype) -> np.ndarray:
    quat = rng.standard_normal((batch_size, 4)).astype(dtype)
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    quat[quat[:, 0] < 0] *= -1
    return quat


def transpose_last_two(x: Array, xp: Any) -> Array:
    return xp.swapaxes(x, -1, -2)


def newton_schulz_project_to_so3(matrix: Array, *, steps: int, xp: Any) -> Array:
    eye = xp.eye(3, dtype=matrix.dtype)
    one = matrix[..., :1, :1] * 0 + 1
    zero = one * 0

    frob = xp.sqrt(xp.sum(matrix * matrix, axis=(-2, -1), keepdims=True))
    safe_frob = xp.where(frob > 0, frob, one)
    x = matrix / safe_frob

    for _ in range(steps):
        xtx = xp.matmul(transpose_last_two(x, xp), x)
        x = 0.5 * xp.matmul(x, 3.0 * eye - xtx)

    det = xp.linalg.det(x)
    sign = xp.where(det < 0, -one[..., 0, 0], one[..., 0, 0])
    correction = xp.stack(
        [
            xp.stack([one[..., 0, 0], zero[..., 0, 0], zero[..., 0, 0]], axis=-1),
            xp.stack([zero[..., 0, 0], one[..., 0, 0], zero[..., 0, 0]], axis=-1),
            xp.stack([zero[..., 0, 0], zero[..., 0, 0], sign], axis=-1),
        ],
        axis=-2,
    )
    return xp.matmul(x, correction)


def zeropower_via_newtonschulz5(matrix: Array, *, steps: int, torch: Any, eps: float = 1e-7) -> Array:
    a, b, c = 3.4445, -4.7750, 2.0315

    x = matrix.to(torch.bfloat16)

    if matrix.shape[-2] > matrix.shape[-1]:
        x = x.mT

    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)

    for _ in range(steps):
        a_mat = x @ x.mT
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x

    if matrix.shape[-2] > matrix.shape[-1]:
        x = x.mT

    return x.to(matrix.dtype)


def project_to_so3_via_newtonschulz5(matrix: Array, *, steps: int, refinement_steps: int, torch: Any, eps: float = 1e-7) -> Array:
    x = zeropower_via_newtonschulz5(matrix, steps=steps, torch=torch, eps=eps).to(torch.float32)
    eye = torch.eye(3, device=matrix.device, dtype=x.dtype)

    for _ in range(refinement_steps):
        x = 0.5 * (x @ (3.0 * eye - x.mT @ x))

    det = torch.linalg.det(x)
    one = torch.ones_like(det)
    sign = torch.where(det < 0, -one, one)
    correction = torch.diag_embed(torch.stack([one, one, sign], dim=-1))
    x = x @ correction

    return x.to(matrix.dtype)


def project_to_so3_newton_schulz_torch(matrix: Array, *, num_iters: int, torch: Any, eps: float = 1e-6) -> Array:
    assert matrix.shape[-2:] == (3, 3), f"Expected (..., 3, 3), got {matrix.shape}"

    eye = torch.eye(3, device=matrix.device, dtype=matrix.dtype).expand_as(matrix)
    frob = torch.linalg.norm(matrix, dim=(-2, -1), keepdim=True).clamp_min(eps)
    x = matrix / frob

    for _ in range(num_iters):
        xtx = x.transpose(-1, -2) @ x
        x = 0.5 * x @ (3.0 * eye - xtx)

    det = torch.det(x)
    col2 = x[..., :, 2]
    x = x.clone()
    x[..., :, 2] = torch.where(det[..., None] < 0, -col2, col2)
    return x


def project_to_so3_via_davenport(matrix: Array, *, steps: int, xp: Any, eps: float = 1e-7) -> Array:
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    trace = m00 + m11 + m22
    z = xp.stack([m21 - m12, m02 - m20, m10 - m01], axis=-1)
    k = xp.stack(
        [
            xp.stack([trace, z[..., 0], z[..., 1], z[..., 2]], axis=-1),
            xp.stack([z[..., 0], m00 - m11 - m22, m01 + m10, m02 + m20], axis=-1),
            xp.stack([z[..., 1], m01 + m10, -m00 + m11 - m22, m12 + m21], axis=-1),
            xp.stack([z[..., 2], m02 + m20, m12 + m21, -m00 - m11 + m22], axis=-1),
        ],
        axis=-2,
    )

    ones = matrix[..., 0, 0] * 0 + 1
    zeros = matrix[..., 0, 0] * 0
    q = xp.stack([ones, zeros, zeros, zeros], axis=-1)

    for _ in range(steps):
        q = xp.matmul(k, q[..., None])[..., 0]
        q = q / (xp.linalg.norm(q, axis=-1, keepdims=True) + eps)

    q = xp.where(q[..., :1] < 0, -q, q)
    return SO3.to_rotmat(q, xp=xp)


def benchmark(fn: Callable[[Array], Array], data: Array, backend: Backend, repeats: int, warmup: int) -> tuple[Array, float, float]:
    for _ in range(warmup):
        backend.sync(fn(data))

    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(data)
        backend.sync(result)
        times.append(time.perf_counter() - start)

    assert result is not None
    return result, float(np.mean(times)), float(np.std(times))


def orthogonality_error(matrix: np.ndarray) -> np.ndarray:
    eye = np.eye(3, dtype=matrix.dtype)
    return np.linalg.norm(np.swapaxes(matrix, -1, -2) @ matrix - eye, axis=(-2, -1))


def determinant_error(matrix: np.ndarray) -> np.ndarray:
    return np.abs(np.linalg.det(matrix) - 1.0)


def print_metrics(
    *,
    noise_level: float,
    batch_size: int,
    results: list[tuple[str, np.ndarray, float, float]],
    clean_matrix: np.ndarray,
) -> None:
    print(f"\nnoise={noise_level:g}, batch={batch_size}")
    svd_label, svd_result, svd_time, _ = results[0]
    assert svd_label == "svd"

    for label, result, mean_time, std_time in results:
        to_clean = np.linalg.norm(result - clean_matrix, axis=(-2, -1)).mean()
        line = (
            f"  {label:<13}"
            f"time={mean_time * 1e3:.3f}ms +- {std_time * 1e3:.3f}ms  "
            f"throughput={batch_size / mean_time:,.0f}/s  "
            f"orth_err={orthogonality_error(result).mean():.3e}  "
            f"det_err={determinant_error(result).mean():.3e}  "
            f"to_clean={to_clean:.3e}"
        )
        if label != "svd":
            to_svd = np.linalg.norm(result - svd_result, axis=(-2, -1)).mean()
            line += f"  to_svd={to_svd:.3e}  speedup={svd_time / mean_time:.2f}x"
        print(line)


def main() -> None:
    args = parse_args()
    dtype = np.dtype(args.dtype)
    rng = np.random.default_rng(args.seed)
    backend = load_backend(args.backend, args.device)

    quat_np = random_quaternions(rng, args.batch_size, dtype)
    clean_matrix = SO3.to_rotmat(backend.asarray(quat_np), xp=backend.xp)
    clean_matrix_np = backend.to_numpy(clean_matrix)

    methods: list[tuple[str, Callable[[Array], Array]]] = [("svd", lambda x: project_to_so3_svd(x, backend.xp))]

    if args.backend == "torch":
        methods.append(
            (
                "ns-simple",
                lambda x: project_to_so3_newton_schulz_torch(
                    x,
                    num_iters=args.steps,
                    torch=backend.xp,
                ),
            )
        )
        methods.append(
            (
                "ns5-so3",
                lambda x: project_to_so3_via_newtonschulz5(
                    x,
                    steps=args.steps,
                    refinement_steps=args.refinement_steps,
                    torch=backend.xp,
                ),
            )
        )
    else:
        methods.append(("newton-schulz", lambda x: newton_schulz_project_to_so3(x, steps=args.steps, xp=backend.xp)))

    methods.append(("davenport", lambda x: project_to_so3_via_davenport(x, steps=args.steps, xp=backend.xp)))

    if args.backend == "jax" and args.jit:
        assert backend.jax is not None
        methods = [(label, backend.jax.jit(fn)) for label, fn in methods]
        for _, fn in methods:
            backend.sync(fn(clean_matrix))

    print(
        f"backend={args.backend} dtype={args.dtype} batch_size={args.batch_size} "
        f"steps={args.steps} refinement_steps={args.refinement_steps}"
    )
    if args.backend == "torch":
        print(f"device={backend.device}")
    print("methods=" + ", ".join(label for label, _ in methods))

    for noise_level in args.noise_levels:
        noise_np = rng.standard_normal((args.batch_size, 3, 3)).astype(dtype)
        noisy_matrix = backend.asarray(clean_matrix_np + noise_level * noise_np)

        results = []
        for label, fn in methods:
            result, mean_time, std_time = benchmark(fn, noisy_matrix, backend, args.repeats, args.warmup)
            results.append((label, backend.to_numpy(result), mean_time, std_time))

        print_metrics(
            noise_level=noise_level,
            batch_size=args.batch_size,
            results=results,
            clean_matrix=clean_matrix_np,
        )


if __name__ == "__main__":
    main()
