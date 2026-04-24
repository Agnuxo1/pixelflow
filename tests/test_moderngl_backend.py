"""Tests for the moderngl GPU backend.

Requires moderngl to be installed (``pip install pixelflow[gpu]``).
Tests are skipped automatically when moderngl is unavailable or when a GPU
context cannot be created on the current machine.
"""

from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl", reason="moderngl not installed — skipping GPU tests")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpu_available() -> bool:
    """Return True if a standalone moderngl context can be created."""
    try:
        ctx = moderngl.create_standalone_context()
        ctx.release()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _small_state(seed: int = 0) -> np.ndarray:
    """Return a random (8, 8, 4) float32 state in [0, 1]."""
    rng = np.random.default_rng(seed)
    return rng.random((8, 8, 4), dtype=np.float32).astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cpu(initial_state: np.ndarray, rule_name: str, steps: int) -> np.ndarray:
    from pixelflow.backends.cpu import run_cpu
    from pixelflow.core.rules import get_rule
    rule = get_rule(rule_name)
    rng = np.random.default_rng(0)
    return run_cpu(initial_state, rule, steps, rng)


def _run_gpu(initial_state: np.ndarray, rule_name: str, steps: int) -> np.ndarray:
    from pixelflow.backends.moderngl_backend import run_moderngl
    from pixelflow.core.rules import get_rule
    rule = get_rule(rule_name)
    rng = np.random.default_rng(0)
    return run_moderngl(initial_state, rule, steps, rng)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule_name", ["diffusion_reaction", "life_like", "wave"])
def test_cpu_vs_gpu_close(gpu_available, rule_name):
    """CPU and GPU outputs should agree within 1e-2 (FP32 GPU vs FP32/FP64 CPU)."""
    if not gpu_available:
        pytest.skip("No GPU context available on this machine")

    state = _small_state(seed=7)
    steps = 5

    cpu_out = _run_cpu(state, rule_name, steps)
    gpu_out = _run_gpu(state, rule_name, steps)

    assert cpu_out.shape == gpu_out.shape == (8, 8, 4)
    diff = np.max(np.abs(cpu_out.astype(np.float32) - gpu_out.astype(np.float32)))
    # Use 1e-3 if both paths are float32; relax to 1e-2 as a safety margin.
    assert diff < 1e-2, (
        f"rule={rule_name}: max abs diff CPU vs GPU = {diff:.6f} (threshold 1e-2)"
    )


@pytest.mark.parametrize("rule_name", ["diffusion_reaction", "life_like", "wave"])
def test_gpu_determinism(gpu_available, rule_name):
    """Two GPU runs with identical inputs must produce bitwise-identical outputs."""
    if not gpu_available:
        pytest.skip("No GPU context available on this machine")

    state = _small_state(seed=13)
    steps = 5
    rng = np.random.default_rng(0)

    from pixelflow.backends.moderngl_backend import run_moderngl
    from pixelflow.core.rules import get_rule
    rule = get_rule(rule_name)

    out1 = run_moderngl(state, rule, steps, rng)
    out2 = run_moderngl(state, rule, steps, rng)

    np.testing.assert_array_equal(
        out1, out2,
        err_msg=f"rule={rule_name}: GPU outputs are not bitwise identical across two runs",
    )


def test_run_moderngl_import_error_without_moderngl(monkeypatch):
    """If moderngl is not importable, run_moderngl should raise ImportError."""
    import sys
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "moderngl":
            raise ImportError("moderngl blocked for test")
        return real_import(name, *args, **kwargs)

    # Remove moderngl from the module cache so the lazy import fires.
    real_mod = sys.modules.pop("moderngl", None)
    try:
        monkeypatch.setattr(builtins, "__import__", _blocked_import)

        from pixelflow.backends.moderngl_backend import run_moderngl
        from pixelflow.core.rules import get_rule

        rule = get_rule("wave")
        state = _small_state()
        rng = np.random.default_rng(0)

        with pytest.raises(ImportError, match="pip install pixelflow\\[gpu\\]"):
            run_moderngl(state, rule, 1, rng)
    finally:
        if real_mod is not None:
            sys.modules["moderngl"] = real_mod


def test_run_moderngl_bad_shape(gpu_available):
    """run_moderngl should raise ValueError for non-(H,W,4) input."""
    if not gpu_available:
        pytest.skip("No GPU context available on this machine")

    from pixelflow.backends.moderngl_backend import run_moderngl
    from pixelflow.core.rules import get_rule
    rule = get_rule("wave")
    rng = np.random.default_rng(0)
    bad_state = np.zeros((8, 8, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        run_moderngl(bad_state, rule, 1, rng)
