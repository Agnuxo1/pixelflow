"""Check paired splits and benchmark summary, without relying on accuracy thresholds."""

import numpy as np
from sklearn.datasets import make_classification

from benchmarks.reproducible_digits import digest, evaluate, summarize
from pixelflow import ReservoirConfig


def test_paired_protocol():
    X, y = make_classification(n_samples=60, n_features=5, random_state=0)
    config = ReservoirConfig(width=4, height=4, steps=1, input_encoding="project")
    rows = evaluate(X, y, [0, 1], config)
    assert len(rows) == 6
    for seed in (0, 1):
        paired = [r for r in rows if r["seed"] == seed]
        assert len({r["train_indices_sha256"] for r in paired}) == 1
        assert len({r["test_indices_sha256"] for r in paired}) == 1
        assert all(r["n_train"] == 45 and r["n_test"] == 15 for r in paired)
    summary = summarize(rows)
    assert summary["raw"]["splits"] == 2
    assert len(summary["reservoir_minus_raw"]["paired_differences"]) == 2


def test_digest_records_shape_and_values():
    X = np.arange(6)
    assert digest(X) == digest(X.copy())
    assert digest(X) != digest(X.reshape(2, 3))
    assert digest(X) != digest(X + 1)
