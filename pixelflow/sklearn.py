"""scikit-learn Pipeline and model-selection integration."""

from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from pixelflow import Reservoir, ReservoirConfig


class ReservoirTransformer(TransformerMixin, BaseEstimator):
    """Fixed reservoir features for two-dimensional tabular input.

    Parameters
    ----------
    config : ReservoirConfig or None
        Reservoir settings. None uses ReservoirConfig defaults. The settings
        are copied at fit time; fit never learns from validation/test rows.
    backend : str, default="cpu"
        Execution backend: cpu, moderngl or cuda.

    Attributes
    ----------
    n_features_in_ : int
        Input column count recorded by fit.
    reservoir_ : Reservoir
        Fitted feature map. Recreated on each fit.
    """

    def __init__(self, config=None, backend="cpu"):
        self.config = config
        self.backend = backend

    def fit(self, X, y=None):
        """Validate input and initialize the fixed feature map."""
        X = check_array(X, dtype=np.float32)
        config = ReservoirConfig() if self.config is None else deepcopy(self.config)
        reservoir = Reservoir(config, backend=self.backend)
        self.n_features_in_ = X.shape[1]
        self.reservoir_ = reservoir
        return self

    def transform(self, X):
        """Return one feature row per input row, preserving row order."""
        check_is_fitted(self, ["reservoir_", "n_features_in_"])
        X = check_array(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but ReservoirTransformer "
                f"is expecting {self.n_features_in_} features as input."
            )
        return self.reservoir_.transform(X)
