"""A feature map must not depend on row order or batch boundaries."""

import numpy as np
import pytest

from pixelflow import Reservoir, ReservoirConfig


@pytest.mark.parametrize("encoding", ["tile", "phase", "project"])
@pytest.mark.parametrize("rule,params", [
    ("diffusion_reaction", {}), ("wave", {}), ("life_like", {"noise": 0.1}),
])
def test_batch_invariance(encoding, rule, params):
    reservoir = Reservoir(ReservoirConfig(
        width=4, height=4, steps=3, seed=7, input_encoding=encoding,
        rule=rule, rule_params=params,
    ))
    X = np.random.default_rng(1).random((6, 5))
    together = reservoir.transform(X)
    separately = np.concatenate([reservoir.transform(row) for row in X])
    np.testing.assert_array_equal(together, separately)
    order = [5, 2, 0, 4, 1, 3]
    np.testing.assert_array_equal(reservoir.transform(X[order]), together[order])
    repeated = reservoir.transform(np.repeat(X[:1], 3, axis=0))
    np.testing.assert_array_equal(repeated, np.repeat(together[:1], 3, axis=0))


@pytest.mark.parametrize("backend,dependency", [("cuda", "cupy"), ("moderngl", "moderngl")])
def test_gpu_projection_batch_invariance(backend, dependency):
    pytest.importorskip(dependency)
    reservoir = Reservoir(ReservoirConfig(
        width=4, height=4, steps=2, seed=7, input_encoding="project",
    ), backend=backend)
    X = np.random.default_rng(1).random((4, 5))
    features = reservoir.transform(X)
    np.testing.assert_allclose(
        features, np.concatenate([reservoir.transform(row) for row in X]), atol=1e-6,
    )
    np.testing.assert_allclose(reservoir.transform(X[::-1]), features[::-1], atol=1e-6)
