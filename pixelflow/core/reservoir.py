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
        elif backend == "cuda":
            try:
                import cupy  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "cuda backend requires cupy to be installed. "
                    "Install with: pip install cupy-cuda12x  "
                    "(or cupy-cuda13x if your toolkit is CUDA 13)"
                ) from exc
        elif backend != "cpu":
            raise ValueError(
                f"Unknown backend '{backend}'. Choose 'cpu', 'moderngl', or 'cuda'."
            )

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
        """X: (N, D_in) float array. Returns (N, feature_dim) float32 features.

        For the ``cuda`` backend all N samples are encoded on CPU first (each
        with its own deterministic RNG seed), then the full (N, H, W, C) batch
        is transferred to GPU in one shot, evolved for ``steps`` iterations,
        and copied back once.  This amortises host<->device transfer overhead
        and lets cupy process all N samples with a single set of kernel
        launches per step.

        For ``cpu`` and ``moderngl`` backends the original per-sample loop is
        preserved (no benefit from batching on those backends).
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[np.newaxis, :]

        c = self.config
        N = len(X)
        features = np.empty((N, self.feature_dim), dtype=np.float32)

        if self.backend == "cuda":
            from pixelflow.backends.cuda_backend import run_cuda_batch

            # Phase 1: encode all samples on CPU (preserves per-sample seeding).
            initial_states = np.empty(
                (N, c.height, c.width, c.channels), dtype=np.float32
            )
            for i, x in enumerate(X):
                rng_enc = np.random.default_rng([c.seed, i])
                initial_states[i] = self._encoder(
                    x, c.height, c.width, c.channels, rng_enc
                )

            # Phase 2: run all steps on GPU as a single 4D batch.
            # rng passed for API parity; GPU path does not use it.
            rng_evo = np.random.default_rng([c.seed, 0, 1])
            finals = run_cuda_batch(
                initial_states, self._rule, c.steps, rng_evo,
                rule_params=c.rule_params,
            )  # (N, H, W, C)

            for i in range(N):
                features[i] = finals[i].ravel()

            return features

        # --- CPU / moderngl: original per-sample loop ---
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
