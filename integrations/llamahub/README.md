# PixelflowReservoirPack

**Experimental research integration.** A LlamaHub pack that wraps a
[pixelflow](https://github.com/franciscoangulo/pixelflow) GPU-texture
reservoir as a feature extractor you can bolt onto any LlamaIndex
embedding pipeline. The reservoir lifts dense embeddings into a higher
dimensional, dynamically-evolved feature space; a small linear readout
(e.g. `RidgeReadout`) is then enough to classify.

This is a research-grade integration: treat it as a showcase of
reservoir computing on top of LlamaIndex, not a production component.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
from integrations.llamahub import PixelflowReservoirPack

# pretend these are LlamaIndex embedding outputs
X_train = np.random.randn(100, 384).astype("float32")
y_train = np.random.randint(0, 2, size=100)
X_test  = np.random.randn(20, 384).astype("float32")

pack = PixelflowReservoirPack(
    width=16, height=16, channels=4, steps=5,
    rule="diffusion_reaction", backend="cpu",
)

# Option A: just get reservoir features and plug into your own classifier
H = pack.run(X_test)                  # (20, 16*16*4)

# Option B: fit the built-in RidgeReadout and predict
pack.fit(X_train, y_train)
y_pred = pack.run(X_test, predict=True)

# LlamaHub convention
modules = pack.get_modules()           # {"reservoir": ..., "config": ..., "readout": ...}
```

## Notes

- Works with any LlamaIndex embedding model — just pass the dense vectors.
- For large batches use `backend="cuda"` or `backend="moderngl"`.
- Feature dimension is `width * height * channels`.
