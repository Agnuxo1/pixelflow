"""PixelflowReservoirPack: wrap a pixelflow Reservoir as a LlamaHub pack.

Experimental research integration — exposes a reservoir-computing feature
extractor that can be used on top of any LlamaIndex embedding pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from llama_index.core.llama_pack.base import BaseLlamaPack

from pixelflow import Reservoir, ReservoirConfig, RidgeReadout


class PixelflowReservoirPack(BaseLlamaPack):
    """Reservoir-computing feature extractor as a LlamaHub pack.

    Parameters
    ----------
    width, height, channels, steps, rule, input_encoding, seed :
        Forwarded to :class:`pixelflow.ReservoirConfig`.
    backend :
        pixelflow backend: ``"cpu"``, ``"moderngl"``, or ``"cuda"``.
    readout :
        Optional fitted/unfitted :class:`pixelflow.RidgeReadout`. If provided,
        :meth:`run` will return predictions instead of raw features when
        ``y`` is supplied to :meth:`fit` first.
    """

    def __init__(
        self,
        width: int = 16,
        height: int = 16,
        channels: int = 4,
        steps: int = 5,
        rule: str = "diffusion_reaction",
        input_encoding: str = "tile",
        seed: int = 0,
        backend: str = "cpu",
        readout: Optional[RidgeReadout] = None,
    ) -> None:
        self.config = ReservoirConfig(
            width=width,
            height=height,
            channels=channels,
            steps=steps,
            rule=rule,
            input_encoding=input_encoding,
            seed=seed,
        )
        self.reservoir = Reservoir(self.config, backend=backend)
        self.readout = readout

    # ------------------------------------------------------------------
    # LlamaHub convention
    # ------------------------------------------------------------------
    def get_modules(self) -> Dict[str, Any]:
        """Expose the underlying objects for inspection / reuse."""
        return {
            "reservoir": self.reservoir,
            "config": self.config,
            "readout": self.readout,
        }

    def fit(self, X, y) -> "PixelflowReservoirPack":
        """Fit the optional RidgeReadout on reservoir features."""
        if self.readout is None:
            self.readout = RidgeReadout(alpha=1.0, task="classification")
        H = self.reservoir.transform(np.asarray(X, dtype=np.float32))
        self.readout.fit(H, np.asarray(y))
        return self

    def run(self, features, predict: bool = False):
        """Transform input features (N, D) through the reservoir.

        Parameters
        ----------
        features : array-like of shape (N, D) or (D,)
            Text/image embeddings, or any dense vectors.
        predict : bool
            If True and a fitted readout exists, return predictions instead
            of the raw reservoir feature matrix.

        Returns
        -------
        np.ndarray
            ``(N, feature_dim)`` reservoir features, or ``(N,)`` predictions.
        """
        X = np.asarray(features, dtype=np.float32)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        H = self.reservoir.transform(X)
        if predict and self.readout is not None:
            return self.readout.predict(H)
        return H
