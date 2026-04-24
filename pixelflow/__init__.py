"""pixelflow: GPU Texture Reservoir Computing.

A minimal, honest library for using fragment-shader cellular automata on GPU textures
as a fixed random reservoir, with a trainable linear readout for downstream tasks
(image classification, time-series, PDE solving).

Author: Francisco Angulo de Lafuente
License: Apache-2.0
"""

from pixelflow._version import __version__
from pixelflow.core.reservoir import Reservoir, ReservoirConfig
from pixelflow.readouts.linear import RidgeReadout, LogisticReadout

__all__ = [
    "__version__",
    "Reservoir",
    "ReservoirConfig",
    "RidgeReadout",
    "LogisticReadout",
]
