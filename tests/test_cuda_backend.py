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


# ---------------------------------------------------------------------------
# Batched mode tests (v0.3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule_name", ["diffusion_reaction", "life_like", "wave"])
def test_batch_matches_per_sample(rule_name: str) -> None:
    """run_cuda_batch on N=4 must match 4 separate run_cuda calls (max diff < 1e-5)."""
    from pixelflow.backends.cuda_backend import run_cuda, run_cuda_batch
    from pixelflow.core.rules import get_rule

    rule  = get_rule(rule_name)
    rng   = np.random.default_rng(99)
    steps = 3

    batch = rng.random((4, 8, 8, 4)).astype(np.float32)

    batched_out = run_cuda_batch(batch, rule, steps, rng)

    for i in range(4):
        per_sample_out = run_cuda(batch[i], rule, steps, rng)
        diff = np.max(np.abs(batched_out[i] - per_sample_out))
        assert diff < 1e-5, (
            f"rule={rule_name} sample {i}: batched vs per-sample max diff = {diff:.2e}"
        )


@pytest.mark.parametrize("rule_name", ["diffusion_reaction", "wave"])
def test_batch_cpu_parity(rule_name: str) -> None:
    """Reservoir.transform with cuda backend must match cpu backend (max diff < 1e-3)."""
    from pixelflow.core.reservoir import Reservoir, ReservoirConfig

    cfg = ReservoirConfig(
        width=8, height=8, channels=4, steps=3,
        rule=rule_name, seed=7,
    )
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 16)).astype(np.float32)

    res_cpu  = Reservoir(cfg, backend="cpu")
    res_cuda = Reservoir(cfg, backend="cuda")

    out_cpu  = res_cpu.transform(X)
    out_cuda = res_cuda.transform(X)

    diff = np.max(np.abs(out_cpu - out_cuda))
    assert diff < 1e-3, (
        f"rule={rule_name}: cpu vs cuda batch max diff = {diff:.6f} (threshold 1e-3)"
    )


def test_batch_determinism() -> None:
    """Two identical run_cuda_batch calls must produce bitwise-identical output."""
    from pixelflow.backends.cuda_backend import run_cuda_batch
    from pixelflow.core.rules import get_rule

    rule  = get_rule("diffusion_reaction")
    rng   = np.random.default_rng(5)
    batch = rng.random((4, 8, 8, 4)).astype(np.float32)

    out1 = run_cuda_batch(batch, rule, 5, rng)
    out2 = run_cuda_batch(batch, rule, 5, rng)

    np.testing.assert_array_equal(out1, out2,
        err_msg="run_cuda_batch is not deterministic across two calls"
    )


def test_run_cuda_batch_bad_shape() -> None:
    """run_cuda_batch must raise ValueError for wrong input shape."""
    from pixelflow.backends.cuda_backend import run_cuda_batch
    from pixelflow.core.rules import get_rule

    rule = get_rule("wave")
    rng  = np.random.default_rng(0)

    with pytest.raises(ValueError):
        run_cuda_batch(np.zeros((4, 8, 8, 3), dtype=np.float32), rule, 1, rng)
