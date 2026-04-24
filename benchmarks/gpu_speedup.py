"""Honest GPU speedup benchmark for pixelflow v0.3.

Compares CPU, moderngl, and batched-CUDA reservoir transforms at configurable
N, grid size, and step count.  Prints a table to stdout and writes JSON to
``benchmarks/results/gpu_speedup_<timestamp>.json``.

Usage::

    python benchmarks/gpu_speedup.py --n 1000 --width 64 --steps 32 --rule diffusion_reaction
    python benchmarks/gpu_speedup.py --n 2000 --width 32 --steps 16 --rule wave

The GPU warmup runs prime the CUDA context and JIT so reported numbers
reflect steady-state throughput, not first-call overhead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

# Ensure the project root is on sys.path when run from any directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pixelflow.core.reservoir import Reservoir, ReservoirConfig


def _time_backend(
    backend: str,
    cfg: ReservoirConfig,
    X: np.ndarray,
    warmup_n: int = 4,
) -> tuple[float, str]:
    """Return (elapsed_seconds, status_note).

    ``status_note`` is empty on success, or a human-readable error message.
    """
    try:
        res = Reservoir(cfg, backend=backend)
    except (ImportError, RuntimeError) as exc:
        return float("nan"), str(exc)

    # Warmup — prime CUDA context / OpenGL context.
    if warmup_n > 0:
        warm_x = X[: min(warmup_n, len(X))]
        try:
            _ = res.transform(warm_x)
        except Exception as exc:  # noqa: BLE001
            return float("nan"), f"warmup failed: {exc}"

    t0 = time.perf_counter()
    try:
        _ = res.transform(X)
    except Exception as exc:  # noqa: BLE001
        return float("nan"), f"transform failed: {exc}"
    return time.perf_counter() - t0, ""


def main() -> int:  # noqa: C901
    parser = argparse.ArgumentParser(description="pixelflow GPU speedup benchmark")
    parser.add_argument("--n",     type=int, default=1000, help="Number of samples")
    parser.add_argument("--width", type=int, default=64,   help="Grid width = height")
    parser.add_argument("--steps", type=int, default=32,   help="CA evolution steps")
    parser.add_argument("--rule",  type=str, default="diffusion_reaction",
                        choices=["diffusion_reaction", "life_like", "wave"])
    parser.add_argument("--d-in",  type=int, default=64,   dest="d_in",
                        help="Input feature dimension")
    args = parser.parse_args()

    N      = args.n
    W      = args.width
    STEPS  = args.steps
    RULE   = args.rule
    D_IN   = args.d_in

    cfg = ReservoirConfig(
        width=W, height=W, channels=4,
        steps=STEPS, rule=RULE,
        input_encoding="tile", seed=0,
    )

    rng = np.random.default_rng(42)
    X   = rng.standard_normal((N, D_IN)).astype(np.float32)

    header = (
        f"pixelflow v0.3 GPU speedup benchmark\n"
        f"  Rule: {RULE}  |  Grid: {W}x{W}  |  Steps: {STEPS}  |  N: {N}\n"
        f"  D_in: {D_IN}  |  Warmup: 4 samples  |  Timer: time.perf_counter"
    )
    print("=" * 60)
    print(header)
    print("=" * 60)

    backends = ["cpu", "moderngl", "cuda"]
    times: dict[str, float] = {}
    notes: dict[str, str]   = {}

    for bk in backends:
        print(f"  [{bk}] running ...", flush=True)
        t, note = _time_backend(bk, cfg, X)
        times[bk] = t
        notes[bk] = note
        if note:
            print(f"  [{bk}] SKIPPED: {note}")
        else:
            ms_per = 1000 * t / N
            print(f"  [{bk}] total {t:.3f} s  ({ms_per:.2f} ms/sample)")

    # Speedup table
    cpu_t = times.get("cpu", float("nan"))
    print()
    print(f"{'Backend':<12} {'Total (s)':>10} {'ms/sample':>12} {'vs CPU':>10}")
    print("-" * 48)
    for bk in backends:
        t = times[bk]
        if np.isnan(t):
            print(f"  {bk:<10} {'N/A':>10} {'N/A':>12} {'N/A':>10}  ({notes[bk][:40]})")
        else:
            ms_per  = 1000 * t / N
            speedup = cpu_t / t if not np.isnan(cpu_t) and t > 0 else float("nan")
            sp_str  = f"{speedup:.2f}x" if not np.isnan(speedup) else "N/A"
            print(f"  {bk:<10} {t:>10.3f} {ms_per:>12.2f} {sp_str:>10}")
    print()

    cuda_t = times.get("cuda", float("nan"))
    if not np.isnan(cuda_t) and not np.isnan(cpu_t):
        ratio = cpu_t / cuda_t
        if ratio >= 2.0:
            print(f"  Result: batched CUDA is {ratio:.1f}x faster than CPU — target met.")
        elif ratio >= 1.0:
            print(
                f"  Result: batched CUDA is {ratio:.2f}x faster than CPU — modest speedup.\n"
                "  At small N or tiny grids the transfer + kernel-launch overhead "
                "limits gains.\n"
                "  Try --n 2000 --width 64 for larger speedup."
            )
        else:
            print(
                f"  Result: batched CUDA is {1/ratio:.2f}x SLOWER than CPU "
                f"(ratio {ratio:.3f}).\n"
                "  Possible causes: very small N, tiny grid, or CUDA context cold-start.\n"
                "  Retry with --n 2000 --width 128 --steps 64 for a fair comparison."
            )

    # Write JSON results
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(_HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"gpu_speedup_{ts}.json")

    payload = {
        "timestamp": ts,
        "config": {
            "rule": RULE, "width": W, "height": W,
            "steps": STEPS, "n": N, "d_in": D_IN,
        },
        "times_s": {bk: (None if np.isnan(t) else round(t, 6))
                    for bk, t in times.items()},
        "ms_per_sample": {
            bk: (None if np.isnan(t) else round(1000 * t / N, 4))
            for bk, t in times.items()
        },
        "speedup_vs_cpu": {
            bk: (None if np.isnan(t) or np.isnan(cpu_t) or t == 0
                 else round(cpu_t / t, 4))
            for bk, t in times.items()
        },
        "notes": notes,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  Results written to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
