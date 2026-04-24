# pixelflow API Contract (v0.1.0)

This document pins down the public API all backends and tasks must implement against.
It is the single source of truth for implementers.

## Core types

### `ReservoirConfig` (dataclass, frozen)

```python
@dataclass(frozen=True)
class ReservoirConfig:
    width: int = 64              # texture width in pixels
    height: int = 64             # texture height in pixels
    channels: int = 4            # RGBA channels (fixed at 4 for GPU compatibility)
    steps: int = 8               # number of CA evolution steps per sample
    rule: str = "diffusion_reaction"  # CA rule name; see pixelflow.core.rules
    rule_params: dict = field(default_factory=dict)  # rule-specific params
    input_encoding: str = "tile"  # how to map input X into initial state
                                  # "tile": tile/resize input into (H, W)
                                  # "phase": encode as phase in channel 0
                                  # "project": random linear projection to (H*W*C)
    seed: int = 0
    dtype: str = "float32"
```

### `Reservoir` (class)

```python
class Reservoir:
    def __init__(self, config: ReservoirConfig, backend: str = "cpu"):
        """backend in {'cpu', 'moderngl'}. Raises ImportError if backend deps missing."""

    @property
    def feature_dim(self) -> int:
        """Return H*W*C (size of flattened feature vector)."""

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D_in) float array. Returns (N, feature_dim) float32 features.
        
        Behavior:
        1. For each sample x in X: encode x into initial state S0 of shape (H, W, C).
        2. Apply CA rule `steps` times deterministically (seed-driven).
        3. Flatten final state to (H*W*C,) feature vector.
        4. Stack all samples.
        
        MUST be deterministic given same config + same X.
        CPU and moderngl backends MUST produce numerically close (not necessarily bitwise)
        results for the same config — tests assert max abs diff < 1e-3 on a synthetic
        fixture.
        """

    def step(self, state: np.ndarray) -> np.ndarray:
        """Apply one CA step to a single state (H, W, C). Useful for visualization."""
```

## CA rules (pixelflow.core.rules)

Each rule is a pure function `(state, params, rng_or_seed) -> next_state` on the CPU side,
and an equivalent GLSL fragment shader for the moderngl backend. Required rules for v0.1:

- **`diffusion_reaction`**: Gray-Scott reaction-diffusion over RGBA. Params: `feed`, `kill`, `dt`, `du`, `dv` with documented defaults that produce stable patterns.
- **`life_like`**: Continuous variant of Conway's Life over channel 0 (majority-vote with tunable threshold). Params: `threshold`, `noise`.
- **`wave`**: Discrete wave equation (two-channel state: amplitude + velocity). Params: `c`, `damping`, `dt`.

Each rule MUST have a CPU numpy implementation AND a GLSL fragment-shader source string.
The GLSL and numpy implementations MUST agree within 1e-3 absolute on the fixture.

## Readouts (pixelflow.readouts.linear)

```python
class RidgeReadout:
    def __init__(self, alpha: float = 1.0): ...
    def fit(self, features: np.ndarray, y: np.ndarray) -> "RidgeReadout": ...
    def predict(self, features: np.ndarray) -> np.ndarray: ...
    def score(self, features: np.ndarray, y: np.ndarray) -> float: ...

class LogisticReadout:
    def __init__(self, C: float = 1.0, max_iter: int = 1000): ...
    # same interface; .predict returns class labels; .predict_proba returns probabilities
```

Both wrap scikit-learn estimators. No custom training loops.

## Tasks (pixelflow.tasks)

- **`mnist.load(subset: int | None = None) -> (X_train, y_train, X_test, y_test)`**
  Returns flattened float32 arrays in [0, 1]. Uses `sklearn.datasets.fetch_openml('mnist_784')`.
- **`eikonal.solve_reference(grid: np.ndarray, source: tuple[int, int]) -> np.ndarray`**
  CPU reference Eikonal solver (fast marching via scipy or hand-rolled) to validate
  the reservoir's ability to learn a geodesic field.
- **`synthetic.two_moons(n: int, noise: float, seed: int) -> (X, y)`**
  For fast-running tests.

## Benchmarks

One end-to-end benchmark script at `benchmarks/mnist_reservoir.py` that:
1. Loads MNIST (default subset=10000 for speed).
2. Trains Reservoir + RidgeReadout.
3. Reports accuracy AND compares against a sklearn baseline (LogisticRegression on raw pixels).
4. Emits a JSON with results and runtime.
5. Does NOT fake numbers. If accuracy is worse than baseline, it reports that honestly.

## Determinism and honesty rules

- No mock data presented as real.
- No `pass` in exception handlers hiding failures.
- If moderngl is not available, `Reservoir(..., backend='moderngl')` must raise `ImportError` with a clear message; do not silently fall back.
- Version is 0.1.0, status is "alpha". Do not claim production-ready.
- Benchmarks report what actually ran, including failures.
