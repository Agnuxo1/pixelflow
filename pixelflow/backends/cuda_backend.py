"""CuPy CUDA backend for pixelflow reservoir computation.

Public API::

    run_cuda(initial_state, rule, steps, rng, rule_params=None) -> np.ndarray
    run_cuda_batch(initial_states, rule, steps, rng, rule_params=None) -> np.ndarray

``run_cuda`` is kept for backwards compatibility; it delegates to
``run_cuda_batch`` with N=1.

``run_cuda_batch`` processes a full (N, H, W, C) tensor on GPU in one shot:
no per-sample host<->device copies, no Python loop over N.  All N samples
share a single set of cupy kernel launches per step via 4D vectorised
expressions.  Axes: axis=0 is N (batch), axis=1 is H (rows), axis=2 is W
(cols), axis=3 is C (channels).  ``cp.roll`` along axis=1 is a row-shift
(equivalent to ``np.roll`` along axis=0 on a single (H,W,C) array), and
``cp.roll`` along axis=2 is a col-shift.
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
# Batched 4D CuPy step functions — input shape (N, H, W, C)
# axis=0 N, axis=1 H (row roll), axis=2 W (col roll), axis=3 C
# ---------------------------------------------------------------------------

def _dr_step_batch(state: "cp.ndarray", params: dict) -> "cp.ndarray":
    """Gray-Scott reaction-diffusion — one step on GPU, batched (N,H,W,C)."""
    feed = float(params.get("feed", 0.055))
    kill = float(params.get("kill", 0.062))
    dt   = float(params.get("dt",   1.0))
    du   = float(params.get("du",   0.16))
    dv   = float(params.get("dv",   0.08))

    U = state[..., 0]
    V = state[..., 1]

    lap_U = (
        cp.roll(U,  1, axis=1) + cp.roll(U, -1, axis=1)
        + cp.roll(U,  1, axis=2) + cp.roll(U, -1, axis=2)
        - 4.0 * U
    )
    lap_V = (
        cp.roll(V,  1, axis=1) + cp.roll(V, -1, axis=1)
        + cp.roll(V,  1, axis=2) + cp.roll(V, -1, axis=2)
        - 4.0 * V
    )

    uvv   = U * V * V
    new_U = cp.clip(U + dt * (du * lap_U - uvv + feed * (1.0 - U)), 0.0, 1.0)
    new_V = cp.clip(V + dt * (dv * lap_V + uvv - (feed + kill) * V), 0.0, 1.0)

    out = state.copy()
    out[..., 0] = new_U
    out[..., 1] = new_V
    return out


def _ll_step_batch(state: "cp.ndarray", params: dict) -> "cp.ndarray":
    """Life-like CA (continuous Conway) — one step on GPU, batched (N,H,W,C).

    Noise is omitted in the GPU path (matches existing per-sample CUDA behaviour
    and the GLSL path).  Tests that compare CPU (with noise>0) to CUDA should
    use tolerance >= 1e-2.
    """
    threshold = float(params.get("threshold", 0.5))

    s     = state[..., 0]
    alive = (s >= threshold).astype(cp.float32)

    n_sum = (
        cp.roll(alive,  1, axis=1) + cp.roll(alive, -1, axis=1)
        + cp.roll(alive,  1, axis=2) + cp.roll(alive, -1, axis=2)
        + cp.roll(cp.roll(alive,  1, axis=1),  1, axis=2)
        + cp.roll(cp.roll(alive,  1, axis=1), -1, axis=2)
        + cp.roll(cp.roll(alive, -1, axis=1),  1, axis=2)
        + cp.roll(cp.roll(alive, -1, axis=1), -1, axis=2)
    )

    born    = ((alive < 0.5) & (n_sum == 3)).astype(cp.float32)
    survive = ((alive >= 0.5) & ((n_sum == 2) | (n_sum == 3))).astype(cp.float32)
    next_s  = cp.clip(born + survive, 0.0, 1.0)

    out = state.copy()
    out[..., 0] = next_s
    return out


def _wv_step_batch(state: "cp.ndarray", params: dict) -> "cp.ndarray":
    """Discrete wave equation — one step on GPU, batched (N,H,W,C)."""
    c       = float(params.get("c",       0.5))
    damping = float(params.get("damping", 0.999))
    dt      = float(params.get("dt",      1.0))

    amp = state[..., 0]
    vel = state[..., 1]

    lap = (
        cp.roll(amp,  1, axis=1) + cp.roll(amp, -1, axis=1)
        + cp.roll(amp,  1, axis=2) + cp.roll(amp, -1, axis=2)
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
# Rule dispatch table (batched 4D)
# ---------------------------------------------------------------------------

_CUDA_BATCH_STEP = {
    "diffusion_reaction": _dr_step_batch,
    "life_like":          _ll_step_batch,
    "wave":               _wv_step_batch,
}

# Per-sample 3D step functions (kept for the single-sample fast path in
# run_cuda when called directly without going through run_cuda_batch).
# These are the original v0.2 functions renamed with _3d suffix.

def _dr_step_cuda(state: "cp.ndarray", params: dict) -> "cp.ndarray":
    """Gray-Scott — one step on GPU, single (H,W,C) array."""
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
    """Life-like CA — one step on GPU, single (H,W,C) array."""
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
    """Discrete wave equation — one step on GPU, single (H,W,C) array."""
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


_CUDA_STEP = {
    "diffusion_reaction": _dr_step_cuda,
    "life_like":          _ll_step_cuda,
    "wave":               _wv_step_cuda,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_cuda_batch(
    initial_states: np.ndarray,
    rule: RuleSpec,
    steps: int,
    rng: np.random.Generator,
    rule_params: dict | None = None,
) -> np.ndarray:
    """Run *rule* for *steps* on a batch of initial states using CuPy.

    All N samples are processed simultaneously as a single 4D tensor on GPU.
    No per-sample device transfers; one ``cp.asnumpy`` call at the end.

    Parameters
    ----------
    initial_states:
        Float32 array of shape (N, H, W, C).
    rule:
        A RuleSpec from ``pixelflow.core.rules``.
    steps:
        Number of CA evolution steps.
    rng:
        Numpy Generator (unused on GPU; present for API parity).
    rule_params:
        Optional parameter overrides for the rule.

    Returns
    -------
    np.ndarray
        Float32 array of shape (N, H, W, C).

    Raises
    ------
    ImportError
        If cupy is not installed.
    ValueError
        If initial_states is not shape (N, H, W, C) with C == 4.
    KeyError
        If rule.name has no CuPy implementation registered.
    """
    _require_cupy()

    if rule_params is None:
        rule_params = {}

    arr = np.ascontiguousarray(initial_states, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[3] != 4:
        raise ValueError(
            f"initial_states must have shape (N, H, W, 4), got {arr.shape}"
        )

    step_fn = _CUDA_BATCH_STEP.get(rule.name)
    if step_fn is None:
        raise KeyError(
            f"No batched CuPy implementation for rule '{rule.name}'. "
            f"Available: {list(_CUDA_BATCH_STEP)}"
        )

    merged = {**rule.default_params, **rule_params}

    state_gpu = cp.asarray(arr)

    for _ in range(steps):
        state_gpu = step_fn(state_gpu, merged)

    return cp.asnumpy(state_gpu).astype(np.float32)


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

    # Delegate to batched path for consistency; unwrap the N=1 result.
    batch_in = state_np[np.newaxis]  # (1, H, W, 4)
    batch_out = run_cuda_batch(batch_in, rule, steps, rng, rule_params=rule_params)
    return batch_out[0]  # (H, W, 4)
