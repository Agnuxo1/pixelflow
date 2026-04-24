"""CIFAR-10 loader for pixelflow benchmarks.

Uses torchvision if installed (default cache location), else downloads the
binary archive via urllib. Returns flattened float32 arrays in [0, 1].
"""
from __future__ import annotations

import logging
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_CACHE_DIR = Path.home() / ".pixelflow_cache" / "cifar10"


def _download_and_extract() -> Path:
    """Download the CIFAR-10 python archive if not cached; return extraction dir."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    extracted = _CACHE_DIR / "cifar-10-batches-py"
    if extracted.exists():
        return extracted
    archive = _CACHE_DIR / "cifar-10-python.tar.gz"
    if not archive.exists():
        logger.info("Downloading CIFAR-10 (~170 MB) to %s ...", archive)
        urllib.request.urlretrieve(_CIFAR_URL, archive)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(_CACHE_DIR)
    return extracted


def _load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        d = pickle.load(f, encoding="latin1")
    X = np.asarray(d["data"], dtype=np.float32).reshape(-1, 3, 32, 32)
    # Convert to (N, 32, 32, 3) then flatten to (N, 3072)
    X = np.transpose(X, (0, 2, 3, 1)).reshape(-1, 32 * 32 * 3) / 255.0
    y = np.asarray(d["labels"], dtype=np.int64)
    return X, y


def load(
    subset: int | None = None, seed: int = 0, grayscale: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_test, y_test) CIFAR-10 arrays.

    - X arrays are float32 in [0, 1], shape (N, 3072) or (N, 1024) if grayscale.
    - y arrays are int64 class labels 0..9.
    - If subset is given, stratified subsample of TRAIN only (test stays 10k).
    """
    root = _download_and_extract()
    X_tr, y_tr = [], []
    for i in range(1, 6):
        Xb, yb = _load_batch(root / f"data_batch_{i}")
        X_tr.append(Xb)
        y_tr.append(yb)
    X_train = np.concatenate(X_tr)
    y_train = np.concatenate(y_tr)
    X_test, y_test = _load_batch(root / "test_batch")

    if grayscale:
        def to_gray(X: np.ndarray) -> np.ndarray:
            X = X.reshape(-1, 32, 32, 3)
            g = 0.299 * X[..., 0] + 0.587 * X[..., 1] + 0.114 * X[..., 2]
            return g.reshape(-1, 32 * 32).astype(np.float32)
        X_train = to_gray(X_train)
        X_test = to_gray(X_test)

    if subset is not None and subset < len(X_train):
        rng = np.random.default_rng(seed)
        per_class = subset // 10
        idx = []
        for c in range(10):
            cls_idx = np.where(y_train == c)[0]
            chosen = rng.choice(cls_idx, size=min(per_class, len(cls_idx)), replace=False)
            idx.append(chosen)
        idx = np.concatenate(idx)
        rng.shuffle(idx)
        X_train, y_train = X_train[idx], y_train[idx]

    return X_train, y_train, X_test, y_test
