"""Integration tests use real sklearn, no external services or downloads."""

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from pixelflow import ReservoirConfig
from pixelflow.sklearn import ReservoirTransformer


def config(seed=7):
    return ReservoirConfig(width=4, height=4, steps=2, seed=seed, input_encoding="project")


def test_pipeline_grid_search_and_serialization():
    X, y = make_classification(n_samples=40, n_features=5, random_state=0)
    pipeline = make_pipeline(ReservoirTransformer(config()), LogisticRegression(max_iter=500))
    search = GridSearchCV(pipeline, {"reservoirtransformer__config": [config(7), config(8)]}, cv=2)
    search.fit(X, y)
    prediction = search.predict(X)
    assert prediction.shape == y.shape
    np.testing.assert_array_equal(search.predict(X[::-1]), prediction[::-1])
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(search)).predict(X), prediction)


def test_clone_does_not_retain_fitted_state():
    transformer = ReservoirTransformer(config()).fit(np.ones((3, 5)))
    fresh = clone(transformer)
    assert fresh.config == transformer.config
    with pytest.raises(NotFittedError):
        fresh.transform(np.ones((3, 5)))


def test_input_validation_and_refit():
    transformer = ReservoirTransformer(config())
    with pytest.raises(NotFittedError):
        transformer.transform([[1, 2]])
    transformer.fit(np.ones((3, 5)))
    with pytest.raises(ValueError, match="expecting 5 features"):
        transformer.transform(np.ones((3, 6)))
    with pytest.raises(ValueError):
        transformer.transform(np.full((3, 5), np.nan))
    transformer.fit(np.ones((3, 6)))
    assert transformer.transform(np.ones((3, 6))).shape == (3, 64)


def test_config_is_snapshot_at_fit_time():
    settings = config()
    transformer = ReservoirTransformer(settings).fit(np.ones((3, 5)))
    settings.rule_params["feed"] = 0.01
    assert transformer.reservoir_.config.rule_params == {}
