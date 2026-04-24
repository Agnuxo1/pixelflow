"""CPU numpy backend for pixelflow reservoir computation."""

from __future__ import annotations

import numpy as np

from pixelflow.core.rules import RuleSpec


def run_cpu(
    initial_state: np.ndarray,
    rule: RuleSpec,
    steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run CA rule for `steps` on initial_state; returns final state (H, W, C)."""
    state = initial_state.astype(np.float32)
    for _ in range(steps):
        state = rule.step(state, rule.default_params, rng)
    return state


def run_cpu_with_params(
    initial_state: np.ndarray,
    rule: RuleSpec,
    rule_params: dict,
    steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run CA rule with explicit params for `steps`; returns final state (H, W, C)."""
    merged = {**rule.default_params, **rule_params}
    state = initial_state.astype(np.float32)
    for _ in range(steps):
        state = rule.step(state, merged, rng)
    return state
