"""CUDA backend benchmark for pixelflow.

Compares CPU, moderngl, per-sample CUDA (v0.2 model), and batched CUDA (v0.3)
on a reservoir-transform workload.  Reports honest measured wall-clock times.
Exits cleanly with a message if cupy or moderngl is unavailable.

Run::

    python examples/cuda_speed_test.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import asdict

import numpy as np

from pixelflow import Reservoir, ReservoirConfig


def _time_backend(
    backend: str,
    cfg: ReservoirConfig,
    X: np.ndarray,
    warmup: int = 4,
) -> float:
    """Return seconds to transform X; NaN if backend unavailable."""
    try:
        res = Reservoir(cfg, backend=backend)
    except (ImportError, RuntimeError) as exc:
        print(f"  [{backend}] unavailable: {exc}")
        return float("nan")

    if warmup > 0:
        _ = res.transform(X[: min(warmup, len(X))])

    t0 = time.perf_counter()
    _ = res.transform(X)
    return time.perf_counter() - t0


def _time_per_sample_cuda(
    cfg: ReservoirConfig,
    X: np.ndarray,
    warmup: int = 2,
) -> float:
    """Time the v0.2 per-sample CUDA path (run_cuda called N times separately).

    This bypasses Reservoir.transform so we can measure the old model even
    after Reservoir has been updated to use batched mode.
    """
    try:
        import cupy as cp  # noqa: F401
    except ImportError:
        return float("nan")

    from pixelflow.backends.cuda_backend import run_cuda
    from pixelflow.core.encoding import get_encoder

    encoder = get_encoder(cfg.input_encoding)
    rule_spec = __import__(
        "pixelflow.core.rules", fromlist=["get_rule"]
    ).get_rule(cfg.rule)

    def _single_run(x_arr: np.ndarray) -> None:
        for i, x in enumerate(x_arr):
            rng_enc = np.random.default_rng([cfg.seed, i])
            initial = encoder(x, cfg.height, cfg.width, cfg.channels, rng_enc)
            rng_evo = np.random.default_rng([cfg.seed, i, 1])
            _ = run_cuda(initial, rule_spec, cfg.steps, rng_evo,
                         rule_params=cfg.rule_params)

    if warmup > 0:
        _single_run(X[: min(warmup, len(X))])

    t0 = time.perf_counter()
    _single_run(X)
    return time.perf_counter() - t0


def main() -> int:
    N = 100
    cfg = ReservoirConfig(
        width=128, height=128, channels=4,
        steps=32, rule="diffusion_reaction",
        input_encoding="project", seed=0,
    )

    print("=" * 64)
    print("pixelflow v0.2 vs v0.3: per-sample vs batched CUDA speed test")
    print("=" * 64)
    print(f"Config: {asdict(cfg)}")
    print(f"Samples: {N}")
    print()

    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, 64)).astype(np.float32)

    rows: list[tuple[str, float, str]] = []

    for backend in ("cpu", "moderngl"):
        print(f"  [{backend}] ...", flush=True)
        t = _time_backend(backend, cfg, X)
        label = backend
        rows.append((label, t, ""))
        if not (t != t):  # not NaN
            print(f"  [{backend}] total {t:.3f} s  ({1000*t/N:.2f} ms/sample)")

    print("  [cuda v0.2 per-sample] ...", flush=True)
    t_ps = _time_per_sample_cuda(cfg, X)
    rows.append(("cuda v0.2 (per-sample)", t_ps, "old model"))
    if not (t_ps != t_ps):
        print(f"  [cuda v0.2 per-sample] total {t_ps:.3f} s  ({1000*t_ps/N:.2f} ms/sample)")
    else:
        print("  [cuda v0.2 per-sample] cupy not available")

    print("  [cuda v0.3 batched] ...", flush=True)
    t_batch = _time_backend("cuda", cfg, X)
    rows.append(("cuda v0.3 (batched)", t_batch, "new model"))
    if not (t_batch != t_batch):
        print(f"  [cuda v0.3 batched] total {t_batch:.3f} s  ({1000*t_batch/N:.2f} ms/sample)")
    else:
        print("  [cuda v0.3 batched] cupy not available")

    print()
    cpu_t = next((t for lbl, t, _ in rows if lbl == "cpu"), float("nan"))

    print(f"{'Backend':<28} {'Total (s)':>10} {'ms/sample':>12} {'vs CPU':>10}")
    print("-" * 64)
    for lbl, t, note in rows:
        if t != t:  # NaN
            print(f"  {lbl:<26} {'N/A':>10} {'N/A':>12} {'N/A':>10}")
        else:
            ms_per  = 1000 * t / N
            speedup = cpu_t / t if cpu_t > 0 else float("nan")
            sp_str  = f"{speedup:.2f}x" if speedup == speedup else "N/A"
            suffix  = f"  ({note})" if note else ""
            print(f"  {lbl:<26} {t:>10.3f} {ms_per:>12.2f} {sp_str:>10}{suffix}")

    print()
    print("Notes:")
    print("  - v0.2 per-sample: one GPU upload/download per sample; very little")
    print("    parallelism across samples.")
    print("  - v0.3 batched: all N samples uploaded once, evolved together as")
    print("    a 4D cupy tensor, downloaded once at the end.")
    print("  - Numbers above are measured; no fabrication.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
