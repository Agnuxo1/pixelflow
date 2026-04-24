"""Linear readout wrappers around scikit-learn estimators."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier, LogisticRegression


class RidgeReadout:
    """Ridge regression or classification readout.

    Task detection via ``task`` arg:
      - ``'auto'`` (default): infers from y. Integer/boolean dtype → classification,
        float dtype → regression.
      - ``'classification'``: always use RidgeClassifier.
      - ``'regression'``: always use Ridge (one-hot encodes multi-class if needed).

    Parameters
    ----------
    alpha:
        Regularisation strength (passed to Ridge/RidgeClassifier).
    task:
        One of ``'auto'``, ``'classification'``, ``'regression'``.
    """

    def __init__(self, alpha: float = 1.0, task: str = "auto") -> None:
        if task not in ("auto", "classification", "regression"):
            raise ValueError(f"task must be 'auto', 'classification', or 'regression'; got {task!r}")
        self.alpha = alpha
        self.task = task
        self._model: Ridge | RidgeClassifier | None = None
        self._resolved_task: str | None = None

    def _resolve_task(self, y: np.ndarray) -> str:
        if self.task != "auto":
            return self.task
        if np.issubdtype(y.dtype, np.integer) or np.issubdtype(y.dtype, np.bool_):
            return "classification"
        return "regression"

    def fit(self, features: np.ndarray, y: np.ndarray) -> "RidgeReadout":
        """Fit on (N, D) features and labels/targets y."""
        self._resolved_task = self._resolve_task(y)
        if self._resolved_task == "classification":
            self._model = RidgeClassifier(alpha=self.alpha)
        else:
            self._model = Ridge(alpha=self.alpha)
        self._model.fit(features, y)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return predictions (class labels or regression values)."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(features)

    def score(self, features: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy (classification) or R^2 (regression)."""
        if self._model is None:
            raise RuntimeError("Call fit() before score().")
        return float(self._model.score(features, y))


class LogisticReadout:
    """Logistic regression readout.

    Parameters
    ----------
    C:
        Inverse regularisation strength.
    max_iter:
        Maximum number of solver iterations.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        self.C = C
        self.max_iter = max_iter
        self._model: LogisticRegression | None = None

    def fit(self, features: np.ndarray, y: np.ndarray) -> "LogisticReadout":
        """Fit on (N, D) features and integer class labels y."""
        self._model = LogisticRegression(C=self.C, max_iter=self.max_iter)
        self._model.fit(features, y)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return predicted class labels."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return class probabilities (N, n_classes)."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return self._model.predict_proba(features)

    def score(self, features: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        if self._model is None:
            raise RuntimeError("Call fit() before score().")
        return float(self._model.score(features, y))
