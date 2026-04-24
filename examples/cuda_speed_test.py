"""CUDA backend benchmark for pixelflow.

Compares CPU, moderngl, and CUDA (cupy) backends on a reservoir-transform
workload. Reports honest measured wall-clock times. Exits cleanly with a
message if cupy or moderngl is unavailable.

Run:
    python examples/cuda_speed_test.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import asdict

import numpy as np

from pixelflow import Reservoir, ReservoirConfig


def _time_backend(backend: str, cfg: ReservoirConfig, X: np.ndarray, warmup: int = 2) -> float:
    """Return seconds to transform X through a Reservoir with the given backend."""
    try:
        res = Reservoir(cfg, backend=backend)
    except ImportError as exc:
        print(f"  [{backend}] unavailable: {exc}")
        return float("nan")
    except RuntimeError as exc:
        print(f"  [{backend}] context creation failed: {exc}")
        return float("nan")

    # Warmup (JIT / context setup)
    if warmup > 0:
        _ = res.transform(X[: min(len(X), 4)])

    t0 = time.perf_counter()
    _ = res.transform(X)
    return time.perf_counter() - t0


def main() -> int:
    N = 100
    cfg = ReservoirConfig(
        width=128, height=128, channels=4,
        steps=32, rule="diffusion_reaction",
        input_encoding="project", seed=0,
    )
    print("=" * 60)
    print("pixelflow: CPU vs moderngl vs CUDA backend speed test")
    print("=" * 60)
    print(f"Config: {asdict(cfg)}")
    print(f"Samples: {N}")
    print()

    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, 64), dtype=np.float32)

    results: dict[str, float] = {}
    for backend in ("cpu", "moderngl", "cuda"):
        print(f"  [{backend}] ...", flush=True)
        t = _time_backend(backend, cfg, X)
        results[backend] = t
        if np.isnan(t):
            continue
        print(f"  [{backend}] total {t:.3f} s  ({1000 * t / N:.2f} ms/sample)")

    print()
    print("Notes:")
    print("  - Standalone GPU contexts add overhead; per-sample creation is the v0.1/v0.2 model.")
    print("  - Expect GPU advantage at larger grids / bigger batches once batching lands.")
    print("  - Numbers above are measured, not claimed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
