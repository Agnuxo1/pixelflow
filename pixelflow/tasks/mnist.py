"""MNIST dataset loader (via sklearn/OpenML)."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load(
    subset: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST with a standard 60k/10k train/test split.

    Parameters
    ----------
    subset:
        If given, draw a stratified subset of this many samples *from the
        training set* (the test set is always the full 10k).
    seed:
        Random seed used for stratified sampling when subset is given.

    Returns
    -------
    X_train, y_train, X_test, y_test
        X arrays are float32 in [0, 1] with shape (N, 784).
        y arrays are int with shape (N,).
    """
    data = fetch_openml("mnist_784", as_frame=False, parser="auto")
    X = data.data.astype(np.float32) / 255.0  # [0, 1]
    y = data.target.astype(np.intp)

    # Standard split: first 60k train, last 10k test (OpenML order preserves this)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    if subset is not None:
        if subset >= len(X_train):
            pass  # use all
        else:
            X_train, _, y_train, _ = train_test_split(
                X_train,
                y_train,
                train_size=subset,
                stratify=y_train,
                random_state=seed,
            )

    return X_train, y_train, X_test, y_test
