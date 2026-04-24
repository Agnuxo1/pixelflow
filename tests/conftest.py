"""Shared pytest fixtures."""

import numpy as np
import pytest

from pixelflow.tasks.synthetic import two_moons, checkerboard


@pytest.fixture(scope="session")
def moons_dataset():
    """Small two-moons dataset (200 samples)."""
    return two_moons(n=200, noise=0.1, seed=42)


@pytest.fixture(scope="session")
def checkerboard_dataset():
    """4x4 checkerboard dataset."""
    return checkerboard(n_per_side=4, seed=0)


@pytest.fixture(scope="session")
def regression_dataset():
    """Simple 1-D regression fixture: y = 2*x0 + x1 + noise."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5)).astype(np.float32)
    y = (2.0 * X[:, 0] + X[:, 1] + 0.1 * rng.standard_normal(200)).astype(np.float32)
    return X, y
