"""Paired digits evaluation; no downloads, tuning or claims of general superiority.

Run from the installed repository:
python -m benchmarks.reproducible_digits --output benchmarks/results/digits.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from pixelflow import ReservoirConfig
from pixelflow.sklearn import ReservoirTransformer


def digest(array):
    """Hash dtype, shape and bytes so recorded datasets/splits are identifiable."""
    array = np.ascontiguousarray(array)
    prefix = f"{array.dtype}:{array.shape}:".encode()
    return hashlib.sha256(prefix + array.tobytes()).hexdigest()


def evaluate(X, y, seeds, config):
    """Use identical held-out rows and classifiers for three fixed feature maps."""
    rows = []
    indices = np.arange(len(y))
    for seed in seeds:
        train, test = train_test_split(indices, test_size=0.25, stratify=y, random_state=seed)
        for name, steps in [("raw", None), ("projection_only", 0), ("reservoir", config.steps)]:
            transforms = []
            if steps is not None:
                transforms.append(ReservoirTransformer(replace(config, steps=steps, seed=seed)))
            model = make_pipeline(
                *transforms, StandardScaler(),
                LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
            )
            started = time.perf_counter()
            with threadpool_limits(limits=1):
                model.fit(X[train], y[train])
                predicted = model.predict(X[test])
                # Catch regressions where predictions depend on sample position.
                np.testing.assert_array_equal(model.predict(X[test][::-1]), predicted[::-1])
                if transforms:
                    feature_map = model.steps[0][1]
                    selected = X[test[:8]]
                    np.testing.assert_allclose(
                        feature_map.transform(selected),
                        np.concatenate([feature_map.transform(row[None]) for row in selected]),
                        rtol=1e-6, atol=1e-6,
                    )
            rows.append({
                "seed": seed, "model": name,
                "accuracy": float(np.mean(predicted == y[test])),
                "elapsed_s_including_invariance_checks": time.perf_counter() - started,
                "train_indices_sha256": digest(train), "test_indices_sha256": digest(test),
                "predictions_sha256": digest(predicted),
                "n_train": len(train), "n_test": len(test),
                "max_solver_iterations": int(model.steps[-1][1].n_iter_.max()),
            })
    return rows


def summarize(rows):
    """Report descriptive variation, not a CI from overlapping random splits."""
    summary = {}
    for name in ("raw", "projection_only", "reservoir"):
        values = [row["accuracy"] for row in rows if row["model"] == name]
        summary[name] = {"mean_accuracy": float(np.mean(values)),
                         "std_across_splits": float(np.std(values)), "splits": len(values)}
    raw = {r["seed"]: r["accuracy"] for r in rows if r["model"] == "raw"}
    differences = [r["accuracy"] - raw[r["seed"]] for r in rows if r["model"] == "reservoir"]
    summary["reservoir_minus_raw"] = {"paired_differences": differences,
                                      "mean_difference": float(np.mean(differences))}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0
    config = ReservoirConfig(width=8, height=8, channels=4, steps=2,
                             rule="diffusion_reaction", input_encoding="project")
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(["git", "-c", f"safe.directory={root.as_posix()}",
                               "rev-parse", "HEAD"], cwd=root, check=True,
                              capture_output=True, text=True).stdout.strip()
    rows = evaluate(X, y, range(5), config)
    payload = {
        "schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_revision": revision,
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: version(name) for name in ("numpy", "scikit-learn", "scipy")},
        "dataset": "sklearn bundled digits (8x8), NOT MNIST",
        "X_sha256": digest(X), "y_sha256": digest(y), "config": asdict(config),
        "protocol": "5 fixed stratified 75/25 splits; no tuning; scaler fit on training only; CPU",
        "limitations": ["Overlapping splits are not independent replications.",
                        "One small dataset and fixed configuration; not a general superiority test.",
                        "Timing includes validation checks and is not a GPU speed benchmark."],
        "runs": rows, "summary": summarize(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
