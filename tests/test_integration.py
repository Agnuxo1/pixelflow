"""End-to-end smoke test: Reservoir + RidgeReadout on two_moons.

This test is skipped if pixelflow.core.reservoir is not yet available
(parallel agent may not have landed yet). Once the core module is present
the test exercises the full pipeline.
"""

import numpy as np
import pytest

reservoir_mod = pytest.importorskip(
    "pixelflow.core.reservoir",
    reason="pixelflow.core.reservoir not available yet",
)

from pixelflow.core.reservoir import Reservoir, ReservoirConfig  # noqa: E402
from pixelflow.readouts import RidgeReadout  # noqa: E402
from pixelflow.tasks.synthetic import two_moons  # noqa: E402


def test_reservoir_ridge_two_moons():
    """Accuracy > chance (>0.6) on two_moons using a small Reservoir."""
    X, y = two_moons(n=400, noise=0.15, seed=0)
    split = 300

    config = ReservoirConfig(
        width=16,
        height=16,
        channels=4,
        steps=3,
        rule="diffusion_reaction",
        input_encoding="tile",
        seed=0,
    )
    res = Reservoir(config, backend="cpu")

    H_train = res.transform(X[:split])
    H_test = res.transform(X[split:])

    readout = RidgeReadout(alpha=1.0, task="classification")
    readout.fit(H_train, y[:split])
    acc = readout.score(H_test, y[split:])

    assert acc > 0.6, (
        f"Integration test failed: accuracy {acc:.3f} not above chance (0.6). "
        "Check reservoir feature quality or readout config."
    )
