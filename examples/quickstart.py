"""Quickstart: Reservoir + RidgeReadout on two_moons.

Run with:
    python examples/quickstart.py

Requires pixelflow.core.reservoir to be available. If not yet installed,
the script exits with a clear message.
"""

import sys
import numpy as np

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
from pixelflow.tasks.synthetic import two_moons

X, y = two_moons(n=500, noise=0.15, seed=0)
X_train, y_train = X[:400], y[:400]
X_test, y_test = X[400:], y[400:]

print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")

# ------------------------------------------------------------------
# Build Reservoir
# ------------------------------------------------------------------
try:
    from pixelflow.core.reservoir import Reservoir, ReservoirConfig
except ImportError:
    print(
        "\npixelflow.core.reservoir not available yet (parallel agent not done).\n"
        "Falling back to raw features for the quickstart demo.",
        file=sys.stderr,
    )
    # Fall back: use raw 2-D features directly
    H_train, H_test = X_train, X_test
else:
    config = ReservoirConfig(
        width=16,
        height=16,
        channels=4,
        steps=5,
        rule="diffusion_reaction",
        input_encoding="tile",
        seed=0,
    )
    res = Reservoir(config, backend="cpu")
    print(f"Reservoir feature dim: {res.feature_dim}")

    H_train = res.transform(X_train)
    H_test = res.transform(X_test)
    print(f"Feature shape (train): {H_train.shape}")

# ------------------------------------------------------------------
# Fit readout
# ------------------------------------------------------------------
from pixelflow.readouts import RidgeReadout

readout = RidgeReadout(alpha=1.0, task="classification")
readout.fit(H_train, y_train)

acc = readout.score(H_test, y_test)
print(f"\nTest accuracy: {acc:.4f}")

if acc > 0.9:
    print("Excellent fit on two_moons.")
elif acc > 0.7:
    print("Reasonable fit on two_moons.")
else:
    print("Accuracy below expectation -- check reservoir configuration.")
