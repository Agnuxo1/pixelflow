"""Synthetic classification datasets."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_moons


def two_moons(
    n: int = 200,
    noise: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for the two-moons problem.

    Parameters
    ----------
    n:
        Total number of samples.
    noise:
        Standard deviation of Gaussian noise added to each point.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    X: float32 array of shape (n, 2)
    y: int array of shape (n,), values in {0, 1}
    """
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return X.astype(np.float32), y.astype(np.intp)


def checkerboard(
    n_per_side: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for a 2D checkerboard classification fixture.

    The grid spans [0, n_per_side] x [0, n_per_side]. Each cell is
    assigned a class based on parity of (row + col). Points are drawn
    uniformly within each cell, 100 per cell.

    Parameters
    ----------
    n_per_side:
        Number of cells per side of the checkerboard.
    seed:
        Random seed.

    Returns
    -------
    X: float32 array of shape (n_per_side^2 * 100, 2)
    y: int array of shape (n_per_side^2 * 100,), values in {0, 1}
    """
    rng = np.random.default_rng(seed)
    points_per_cell = 100
    X_list, y_list = [], []
    for row in range(n_per_side):
        for col in range(n_per_side):
            pts = rng.uniform(
                low=[col, row],
                high=[col + 1, row + 1],
                size=(points_per_cell, 2),
            )
            label = (row + col) % 2
            X_list.append(pts)
            y_list.append(np.full(points_per_cell, label, dtype=np.intp))
    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list)
    return X, y
