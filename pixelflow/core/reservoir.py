"""Reservoir class and ReservoirConfig dataclass."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from pixelflow.core.rules import get_rule, RuleSpec
from pixelflow.core.encoding import get_encoder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReservoirConfig:
    """Immutable configuration for a Reservoir."""

    width: int = 64
    height: int = 64
    channels: int = 4
    steps: int = 8
    rule: str = "diffusion_reaction"
    rule_params: dict = field(default_factory=dict)
    input_encoding: str = "tile"
    seed: int = 0
    dtype: str = "float32"


class Reservoir:
    """GPU Texture Reservoir: encodes inputs, evolves CA, returns feature maps."""

    def __init__(self, config: ReservoirConfig, backend: str = "cpu") -> None:
        """backend in {'cpu', 'moderngl'}. Raises ImportError if backend deps missing."""
        if backend == "moderngl":
            try:
                import moderngl  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "moderngl backend requires 'moderngl' to be installed. "
                    "Install with: pip install moderngl"
                ) from exc
        elif backend != "cpu":
            raise ValueError(f"Unknown backend '{backend}'. Choose 'cpu' or 'moderngl'.")

        self.config = config
        self.backend = backend
        self._rule: RuleSpec = get_rule(config.rule)
        self._encoder = get_encoder(config.input_encoding)

    @property
    def feature_dim(self) -> int:
        """Return H*W*C (size of flattened feature vector)."""
        c = self.config
        return c.height * c.width * c.channels

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D_in) float array. Returns (N, feature_dim) float32 features."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[np.newaxis, :]

        c = self.config
        features = np.empty((len(X), self.feature_dim), dtype=np.float32)

        if self.backend == "cpu":
            from pixelflow.backends.cpu import run_cpu_with_params
            runner = run_cpu_with_params
        else:
            from pixelflow.backends.moderngl_backend import run_moderngl

            def runner(initial, rule, params, steps, rng):
                return run_moderngl(initial, rule, steps, rng, rule_params=params)

        for i, x in enumerate(X):
            rng = np.random.default_rng([c.seed, i])
            initial = self._encoder(x, c.height, c.width, c.channels, rng)
            rng_evo = np.random.default_rng([c.seed, i, 1])
            final = runner(initial, self._rule, c.rule_params, c.steps, rng_evo)
            features[i] = final.astype(np.float32).ravel()

        return features

    def step(self, state: np.ndarray) -> np.ndarray:
        """Apply one CA step to a single state (H, W, C). Useful for visualization."""
        state = np.asarray(state, dtype=np.float32)
        c = self.config
        merged = {**self._rule.default_params, **c.rule_params}
        rng = np.random.default_rng(c.seed)
        return self._rule.step(state, merged, rng)
