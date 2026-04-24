"""CuPy CUDA backend for pixelflow reservoir computation.

Public API::

    run_cuda(initial_state, rule, steps, rng, rule_params=None) -> np.ndarray

Signature matches run_moderngl exactly (rng present for API parity; GPU path
is deterministic without it).

Implementation strategy — v0.2: Option A (vectorised cupy expressions).
Each rule step is a direct translation of the numpy CPU step: swap np -> cp
and use cupy.roll for periodic boundaries.  This gives identical numerics and
zero custom CUDA C.  RawKernel per-pixel optimisation is v0.3 territory.
"""

from __future__ import annotations

import numpy as np

from pixelflow.core.rules import RuleSpec


# ---------------------------------------------------------------------------
# cupy import — optional dependency
# ---------------------------------------------------------------------------

try:
    import cupy as cp  # type: ignore[import]
except ImportError as _cupy_missing:
    cp = None  # type: ignore[assignment]
    _CUPY_ERR = _cupy_missing
else:
    _CUPY_ERR = None


def _require_cupy() -> None:
    """Raise a clear ImportError if cupy is unavailable."""
    if cp is None:
        raise ImportError(
            "cuda backend requires cupy. "
            "Install with: pip install cupy-cuda12x  "
            "(or cupy-cuda13x if your toolkit is CUDA 13)"
        ) from _CUPY_ERR


# ---------------------------------------------------------------------------
# Per-rule CuPy step functions
# (Option A: np -> cp substitution; behaviour identical to cpu.py / rules.py)
# ---------------------------------------------------------------------------

def _dr_step_cuda(state: "cp.ndarray", params: dict) -> "cp.ndarray":
    """Gray-Scott reaction-diffusion — one step on GPU."""
    feed = float(params.get("feed", 0.055))
    kill = float(params.get("kill", 0.062))
    dt   = float(params.get("dt",   1.0))
    du   = float(params.get("du",   0.16))
    dv   = float(params.get("dv",   0.08))

    U = state[..., 0]
    V = state[..., 1]

    lap_U = (
        cp.roll(U,  1, axis=0) + cp.roll(U, -1, axis=0)
        + cp.roll(U,  1, axis=1) + cp.roll(U, -1, axis=1)
        - 4.0 * U
    )
    lap_V = (
        cp.roll(V,  1, axis=0) + cp.roll(V, -1, axis=0)
        + cp.roll(V,  1, axis=1) + cp.roll(V, -1, axis=1)
        - 4.0 * V
    )

    uvv   = U * V * V
    new_U = cp.clip(U + dt * (du * lap_U - uvv + feed * (1.0 - U)), 0.0, 1.0)
    new_V = cp.clip(V + dt * (dv * lap_V + uvv - (feed + kill) * V), 0.0, 1.0)

    out = state.copy()
    out[..., 0] = new_U
    out[..., 1] = new_V
    return out


def _ll_step_cuda(state: "cp.ndarray", params: dict) -> "cp.ndarray":
    """Life-like CA (continuous Conway) — one step on GPU.

    Note: the noise path is intentionally omitted on the GPU side (mirrors
    the GPU/GLSL behaviour which also drops noise for determinism).
    """
    threshold = float(params.get("threshold", 0.5))

    s     = state[..., 0]
    alive = (s >= threshold).astype(cp.float32)

    n_sum = (
        cp.roll(alive,  1, axis=0) + cp.roll(alive, -1, axis=0)
        + cp.roll(alive,  1, axis=1) + cp.roll(alive, -1, axis=1)
        + cp.roll(cp.roll(alive,  1, axis=0),  1, axis=1)
        + cp.roll(cp.roll(alive,  1, axis=0), -1, axis=1)
        + cp.roll(cp.roll(alive, -1, axis=0),  1, axis=1)
        + cp.roll(cp.roll(alive, -1, axis=0), -1, axis=1)
    )

    born    = ((alive < 0.5) & (n_sum == 3)).astype(cp.float32)
    survive = ((alive >= 0.5) & ((n_sum == 2) | (n_sum == 3))).astype(cp.float32)
    next_s  = cp.clip(born + survive, 0.0, 1.0)

    out = state.copy()
    out[..., 0] = next_s
    return out


def _wv_step_cuda(state: "cp.ndarray", params: dict) -> "cp.ndarray":
    """Discrete wave equation — one step on GPU."""
    c       = float(params.get("c",       0.5))
    damping = float(params.get("damping", 0.999))
    dt      = float(params.get("dt",      1.0))

    amp = state[..., 0]
    vel = state[..., 1]

    lap = (
        cp.roll(amp,  1, axis=0) + cp.roll(amp, -1, axis=0)
        + cp.roll(amp,  1, axis=1) + cp.roll(amp, -1, axis=1)
        - 4.0 * amp
    )

    c2      = c * c
    new_vel = (vel + dt * c2 * lap) * damping
    new_amp = cp.clip(amp + dt * new_vel, -1.0, 1.0)

    out = state.copy()
    out[..., 0] = new_amp
    out[..., 1] = new_vel
    return out


# ---------------------------------------------------------------------------
# Rule dispatch table
# ---------------------------------------------------------------------------

_CUDA_STEP = {
    "diffusion_reaction": _dr_step_cuda,
    "life_like":          _ll_step_cuda,
    "wave":               _wv_step_cuda,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_cuda(
    initial_state: np.ndarray,
    rule: RuleSpec,
    steps: int,
    rng: np.random.Generator,
    rule_params: dict | None = None,
) -> np.ndarray:
    """Run *rule* for *steps* on *initial_state* using the CuPy CUDA backend.

    Parameters
    ----------
    initial_state:
        Float32 array of shape (H, W, 4).
    rule:
        A RuleSpec from ``pixelflow.core.rules``.
    steps:
        Number of CA evolution steps.
    rng:
        A numpy random Generator (unused on GPU; present for API parity with
        the CPU backend).
    rule_params:
        Optional parameter overrides for the rule (merged with defaults).

    Returns
    -------
    np.ndarray
        Float32 array of shape (H, W, 4) — the final CA state.

    Raises
    ------
    ImportError
        If cupy is not installed.
    ValueError
        If initial_state is not shape (H, W, 4).
    KeyError
        If rule.name has no CuPy implementation registered.
    """
    _require_cupy()

    if rule_params is None:
        rule_params = {}

    state_np = np.ascontiguousarray(initial_state, dtype=np.float32)
    if state_np.ndim != 3 or state_np.shape[2] != 4:
        raise ValueError(
            f"initial_state must have shape (H, W, 4), got {state_np.shape}"
        )

    step_fn = _CUDA_STEP.get(rule.name)
    if step_fn is None:
        raise KeyError(
            f"No CuPy implementation for rule '{rule.name}'. "
            f"Available: {list(_CUDA_STEP)}"
        )

    merged = {**rule.default_params, **rule_params}

    # Transfer to GPU
    state_gpu = cp.asarray(state_np)

    for _ in range(steps):
        state_gpu = step_fn(state_gpu, merged)

    # Transfer back to CPU
    return cp.asnumpy(state_gpu).astype(np.float32)
