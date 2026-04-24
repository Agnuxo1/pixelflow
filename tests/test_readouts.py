"""Unit tests for pixelflow.readouts."""

import numpy as np
import pytest

from pixelflow.readouts import RidgeReadout, LogisticReadout


class TestRidgeReadout:
    def test_regression_auto(self, regression_dataset):
        X, y = regression_dataset
        split = len(X) // 2
        model = RidgeReadout(alpha=1.0)
        model.fit(X[:split], y[:split])
        r2 = model.score(X[split:], y[split:])
        assert r2 > 0.8, f"R^2 too low: {r2}"

    def test_regression_explicit(self, regression_dataset):
        X, y = regression_dataset
        model = RidgeReadout(alpha=1.0, task="regression")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape

    def test_classification_auto(self, moons_dataset):
        X, y = moons_dataset
        split = 100
        model = RidgeReadout(alpha=1.0)
        # int labels → auto-detected as classification
        model.fit(X[:split], y[:split])
        acc = model.score(X[split:], y[split:])
        assert acc > 0.8, f"accuracy too low: {acc}"

    def test_classification_explicit(self, moons_dataset):
        X, y = moons_dataset
        model = RidgeReadout(alpha=1.0, task="classification")
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_before_fit_raises(self):
        model = RidgeReadout()
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((5, 3)))

    def test_score_before_fit_raises(self):
        model = RidgeReadout()
        with pytest.raises(RuntimeError):
            model.score(np.zeros((5, 3)), np.zeros(5))

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            RidgeReadout(task="bad_task")

    def test_single_class_behaviour(self):
        """Single-class y: RidgeClassifier fits without raising (sklearn behaviour).

        sklearn's RidgeClassifier accepts single-class inputs silently and always
        predicts that one class. We verify the model at least returns the correct
        constant prediction rather than crashing or returning garbage.
        """
        X = np.random.default_rng(0).standard_normal((20, 4)).astype(np.float32)
        y = np.zeros(20, dtype=np.intp)  # all same class
        model = RidgeReadout(task="classification")
        model.fit(X, y)
        preds = model.predict(X)
        # All predictions should be the single class value
        assert np.all(preds == 0), f"unexpected predictions: {preds}"


class TestLogisticReadout:
    def test_two_moons_accuracy(self):
        from pixelflow.tasks.synthetic import two_moons
        X, y = two_moons(n=500, noise=0.1, seed=0)
        split = 400
        model = LogisticReadout(C=1.0, max_iter=1000)
        model.fit(X[:split], y[:split])
        acc = model.score(X[split:], y[split:])
        assert acc > 0.9, f"accuracy too low: {acc}"

    def test_predict_proba_shape(self, moons_dataset):
        X, y = moons_dataset
        model = LogisticReadout()
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_before_fit_raises(self):
        model = LogisticReadout()
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((5, 3)))

    def test_predict_proba_before_fit_raises(self):
        model = LogisticReadout()
        with pytest.raises(RuntimeError):
            model.predict_proba(np.zeros((5, 3)))
