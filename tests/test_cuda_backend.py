"""Tests for the CuPy CUDA backend.

Requires cupy to be installed (``pip install pixelflow[cuda]``).
Tests are skipped automatically when cupy is unavailable.
"""

from __future__ import annotations

import sys
import builtins

import numpy as np
import pytest

cupy = pytest.importorskip("cupy", reason="cupy not installed — skipping CUDA tests")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _small_state(seed: int = 0) -> np.ndarray:
    """Return a random (8, 8, 4) float32 state in [0, 1]."""
    rng = np.random.default_rng(seed)
    return rng.random((8, 8, 4)).astype(np.float32)


def _run_cpu(initial_state: np.ndarray, rule_name: str, steps: int) -> np.ndarray:
    from pixelflow.backends.cpu import run_cpu
    from pixelflow.core.rules import get_rule
    rule = get_rule(rule_name)
    rng = np.random.default_rng(0)
    return run_cpu(initial_state, rule, steps, rng)


def _run_cuda(initial_state: np.ndarray, rule_name: str, steps: int) -> np.ndarray:
    from pixelflow.backends.cuda_backend import run_cuda
    from pixelflow.core.rules import get_rule
    rule = get_rule(rule_name)
    rng = np.random.default_rng(0)
    return run_cuda(initial_state, rule, steps, rng)


# ---------------------------------------------------------------------------
# Parity tests: CPU vs CUDA
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule_name", ["diffusion_reaction", "life_like", "wave"])
def test_cpu_vs_cuda_close(rule_name: str) -> None:
    """CPU and CUDA outputs must agree within 1e-3 on an 8x8x4 fixture, 5 steps."""
    state = _small_state(seed=7)
    steps = 5

    cpu_out  = _run_cpu(state, rule_name, steps)
    cuda_out = _run_cuda(state, rule_name, steps)

    assert cpu_out.shape == cuda_out.shape == (8, 8, 4)
    diff = np.max(np.abs(cpu_out.astype(np.float32) - cuda_out.astype(np.float32)))
    assert diff < 1e-3, (
        f"rule={rule_name}: max abs diff CPU vs CUDA = {diff:.6f} (threshold 1e-3)"
    )


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule_name", ["diffusion_reaction", "life_like", "wave"])
def test_cuda_determinism(rule_name: str) -> None:
    """Two CUDA runs with identical inputs must produce identical outputs."""
    from pixelflow.backends.cuda_backend import run_cuda
    from pixelflow.core.rules import get_rule

    state = _small_state(seed=13)
    steps = 5
    rule  = get_rule(rule_name)
    rng   = np.random.default_rng(0)

    out1 = run_cuda(state, rule, steps, rng)
    out2 = run_cuda(state, rule, steps, rng)

    np.testing.assert_array_equal(
        out1, out2,
        err_msg=f"rule={rule_name}: CUDA outputs are not identical across two runs",
    )


# ---------------------------------------------------------------------------
# ImportError when cupy is absent
# ---------------------------------------------------------------------------

def test_run_cuda_import_error_without_cupy(monkeypatch: pytest.MonkeyPatch) -> None:
    """If cupy is not importable, run_cuda must raise ImportError with pip hint."""
    real_import = builtins.__import__

    def _blocked_import(name: str, *args, **kwargs):
        if name == "cupy":
            raise ImportError("cupy blocked for test")
        return real_import(name, *args, **kwargs)

    # Remove cupy from module cache so the lazy check inside cuda_backend fires.
    real_cupy = sys.modules.pop("cupy", None)
    # Also reload cuda_backend so its module-level cp = None path is exercised.
    real_cuda_mod = sys.modules.pop("pixelflow.backends.cuda_backend", None)

    try:
        monkeypatch.setattr(builtins, "__import__", _blocked_import)

        import importlib
        import pixelflow.backends.cuda_backend as _mod  # noqa: F401 — fresh import

        # Force reimport with blocked cupy
        if "pixelflow.backends.cuda_backend" in sys.modules:
            del sys.modules["pixelflow.backends.cuda_backend"]

        from pixelflow.core.rules import get_rule
        rule  = get_rule("wave")
        state = _small_state()
        rng   = np.random.default_rng(0)

        # Import fresh module (cupy will fail to import at module level).
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "pixelflow.backends.cuda_backend_test_isolated",
            "D:/PROJECTS/pixelflow/pixelflow/backends/cuda_backend.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        with pytest.raises(ImportError, match="pip install cupy"):
            mod.run_cuda(state, rule, 1, rng)

    finally:
        if real_cupy is not None:
            sys.modules["cupy"] = real_cupy
        if real_cuda_mod is not None:
            sys.modules["pixelflow.backends.cuda_backend"] = real_cuda_mod


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------

def test_run_cuda_bad_shape() -> None:
    """run_cuda must raise ValueError for non-(H,W,4) input."""
    from pixelflow.backends.cuda_backend import run_cuda
    from pixelflow.core.rules import get_rule

    rule      = get_rule("wave")
    rng       = np.random.default_rng(0)
    bad_state = np.zeros((8, 8, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        run_cuda(bad_state, rule, 1, rng)
