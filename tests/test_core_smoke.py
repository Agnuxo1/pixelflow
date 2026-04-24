"""Smoke tests for pixelflow.core: shape, NaN/Inf checks across all rules."""

import numpy as np
import pytest

from pixelflow.core import Reservoir, ReservoirConfig

RULES = ["diffusion_reaction", "life_like", "wave"]
N_SAMPLES = 5
H, W, C = 4, 4, 4
STEPS = 3
RNG = np.random.default_rng(42)


@pytest.mark.parametrize("rule", RULES)
def test_transform_shape(rule):
    """transform() returns (N, H*W*C) shape."""
    cfg = ReservoirConfig(height=H, width=W, channels=C, steps=STEPS, rule=rule, seed=0)
    res = Reservoir(cfg, backend="cpu")
    X = RNG.random((N_SAMPLES, 16)).astype(np.float32)
    features = res.transform(X)
    assert features.shape == (N_SAMPLES, H * W * C), f"Bad shape {features.shape}"


@pytest.mark.parametrize("rule", RULES)
def test_no_nan_inf(rule):
    """transform() produces no NaN or Inf values."""
    cfg = ReservoirConfig(height=H, width=W, channels=C, steps=STEPS, rule=rule, seed=1)
    res = Reservoir(cfg, backend="cpu")
    X = RNG.random((N_SAMPLES, 16)).astype(np.float32)
    features = res.transform(X)
    assert np.all(np.isfinite(features)), f"NaN/Inf found in {rule} features"


@pytest.mark.parametrize("rule", RULES)
def test_deterministic(rule):
    """transform() is deterministic given the same config and input."""
    cfg = ReservoirConfig(height=H, width=W, channels=C, steps=STEPS, rule=rule, seed=7)
    X = RNG.random((N_SAMPLES, 16)).astype(np.float32)
    f1 = Reservoir(cfg, backend="cpu").transform(X)
    f2 = Reservoir(cfg, backend="cpu").transform(X)
    np.testing.assert_array_equal(f1, f2)


def test_feature_dim():
    """feature_dim property returns H*W*C."""
    cfg = ReservoirConfig(height=H, width=W, channels=C)
    res = Reservoir(cfg)
    assert res.feature_dim == H * W * C


def test_step_shape():
    """step() returns same shape as input state."""
    cfg = ReservoirConfig(height=H, width=W, channels=C, rule="wave")
    res = Reservoir(cfg)
    state = np.random.default_rng(0).random((H, W, C)).astype(np.float32)
    next_state = res.step(state)
    assert next_state.shape == (H, W, C)
