# Integrations

Reservoir-computing doesn't map cleanly onto most LLM tooling, so integrations
here are narrow and honest: thin adapters with smoke tests that actually pass.

## LlamaIndex / LlamaHub adapter

Location: `integrations/llamahub/`

Feature-extractor class `PixelflowReservoirPack` that lifts dense embedding
vectors (e.g. from any LlamaIndex embedding model) into a high-dimensional
reservoir feature space, then optionally trains a `RidgeReadout` on top.

Verified end-to-end (April 2026) against `llama-index-core` main. Smoke test:

```bash
cd integrations/llamahub
python -c "
from base import PixelflowReservoirPack
import numpy as np
pack = PixelflowReservoirPack(width=16, height=16, steps=4, rule='wave')
X = np.random.randn(5, 32).astype('float32')
print(pack.run(X).shape)        # (5, 1024)
pack.fit(X, [0,1,0,1,0])
print(pack.run(X, predict=True)) # [0 1 0 1 0]
"
```

`BaseLlamaPack` was removed from `llama-index-core` 0.12; we kept the
`get_modules()` / `run()` surface and dropped the dead inheritance. No runtime
dependency on llama-index.

## Open WebUI tool

Location: `integrations/open-webui/`

Single-file custom tool (`pixelflow_tool.py`) exposing a `Tools` class with
`classify_with_reservoir(image_base64: str)`. Decodes a base64 image, runs it
through a small CPU reservoir (wave, 32×32×4, 4 steps), returns a one-line
description of the reservoir feature-vector statistics.

Verified end-to-end (April 2026) with a synthetic 28×28 PNG:

```
Reservoir features: dim=4096, L2 norm=16.7727,
sparsity (|v|<0.0001)=0.501. Config: 32x32x4, rule='wave', steps=4.
```

Install in Open WebUI via *Workspace → Tools → Create new tool* and paste the
file contents, or upload the file directly.

## What is **not** integrated (on purpose)

- **LangChain / CrewAI / Haystack** — reservoir features over embeddings is a
  niche trick; a wrapper would be trivial but adds no real value beyond the
  LlamaIndex one. Skipped to avoid padding.
- **awesome-* lists** — the two reservoir-computing ones are stale; the active
  LLM-centric lists (`awesome-local-ai` etc.) are off-topic for pixelflow.
  Submitting anyway would be spam.
- **VS Code / JetBrains / Chrome extensions** — no meaningful UI surface for
  a numeric research library. Will revisit only if a real user-facing
  visualization tool emerges.
