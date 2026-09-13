"""Run with python examples/sklearn_pipeline.py after installing pixelflow-rc."""

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from pixelflow import ReservoirConfig
from pixelflow.sklearn import ReservoirTransformer


def main():
    X, y = make_classification(n_samples=120, n_features=8, random_state=42)
    config = ReservoirConfig(width=4, height=4, steps=2, input_encoding="project", seed=7)
    pipeline = make_pipeline(ReservoirTransformer(config), LogisticRegression(max_iter=500))
    scores = cross_val_score(pipeline, X, y, cv=3)
    print(f"Synthetic-data CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")


if __name__ == "__main__":
    main()
