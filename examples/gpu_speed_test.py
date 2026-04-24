"""GPU vs CPU speed comparison for pixelflow reservoir backends.

Benchmarks both backends on 100 samples of a 128x128 state for 32 steps.
Prints actual wall-clock times measured with time.perf_counter.

NOTE: For small grids (128x128) and few steps, GPU context creation overhead
often makes moderngl *slower* than CPU. This is expected and reported honestly.
Performance cross-over typically occurs for larger grids (512x512+) or when
amortising context creation across many batches.

Usage::

    python examples/gpu_speed_test.py
"""

from __future__ import annotations

import sys
import os
import time

# Ensure the project root is on sys.path when running this script directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_SAMPLES = 100
GRID_H = 128
GRID_W = 128
STEPS = 32
RULE_NAME = "diffusion_reaction"
SEED = 0

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

rng = np.random.default_rng(SEED)
states = [
    rng.random((GRID_H, GRID_W, 4), dtype=np.float32).astype(np.float32)
    for _ in range(N_SAMPLES)
]

print(f"pixelflow GPU vs CPU speed benchmark")
print(f"  Rule   : {RULE_NAME}")
print(f"  Grid   : {GRID_H}x{GRID_W}  |  Steps: {STEPS}  |  Samples: {N_SAMPLES}")
print()

# ---------------------------------------------------------------------------
# CPU benchmark
# ---------------------------------------------------------------------------

try:
    from pixelflow.backends.cpu import run_cpu
    from pixelflow.core.rules import get_rule

    rule = get_rule(RULE_NAME)
    cpu_rng = np.random.default_rng(SEED)

    t0 = time.perf_counter()
    for s in states:
        run_cpu(s, rule, STEPS, cpu_rng)
    cpu_time = time.perf_counter() - t0

    print(f"CPU backend  : {cpu_time:.3f}s total  ({cpu_time / N_SAMPLES * 1000:.2f} ms/sample)")
except Exception as exc:
    print(f"CPU backend  : FAILED — {exc}")
    cpu_time = None

# ---------------------------------------------------------------------------
# moderngl benchmark
# ---------------------------------------------------------------------------

try:
    import moderngl  # noqa: F401  (check availability before timing)
except ImportError:
    print("moderngl     : NOT INSTALLED — run 'pip install pixelflow[gpu]' to enable GPU backend")
    print()
    print("NOTE: Performance cross-over favours GPU for grids >= 512x512 or large batch sizes.")
    sys.exit(0)

try:
    _ctx_test = moderngl.create_standalone_context()
    _ctx_test.release()
except Exception as exc:
    print(f"moderngl     : GPU CONTEXT UNAVAILABLE — {exc}")
    print("  Skipping GPU benchmark. No GPU driver or EGL available.")
    sys.exit(0)

try:
    from pixelflow.backends.moderngl_backend import run_moderngl
    from pixelflow.core.rules import get_rule

    rule = get_rule(RULE_NAME)
    gpu_rng = np.random.default_rng(SEED)

    # Each call creates and destroys a context (v0.1 behaviour — honest overhead).
    t0 = time.perf_counter()
    for s in states:
        run_moderngl(s, rule, STEPS, gpu_rng)
    gpu_time = time.perf_counter() - t0

    print(f"moderngl GPU : {gpu_time:.3f}s total  ({gpu_time / N_SAMPLES * 1000:.2f} ms/sample)")

    if cpu_time is not None:
        ratio = gpu_time / cpu_time
        if ratio > 1.0:
            print(f"\nResult: GPU is {ratio:.1f}x SLOWER than CPU for this workload.")
            print("  This is expected for small grids — context creation dominates.")
            print("  A shared-context v0.2 implementation will amortise this cost.")
        else:
            print(f"\nResult: GPU is {1.0/ratio:.1f}x FASTER than CPU.")

except Exception as exc:
    print(f"moderngl GPU : FAILED — {exc}")

print()
print("NOTE: Performance cross-over favours GPU for grids >= 512x512 or when")
print("      context creation is amortised across many samples (planned for v0.2).")
