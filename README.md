# pixelflow

**GPU Texture Reservoir Computing.** A minimal, honest library that treats image
pixels on a GPU texture as a fixed random reservoir — evolving under local
cellular-automaton rules expressed as fragment shaders — and trains a linear
readout on top for image classification, PDE solving, and time-series tasks.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-alpha-orange)]()
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/Agnuxo1/pixelflow)

> **Live demo:** <https://huggingface.co/spaces/Agnuxo1/pixelflow>

> **Status: v0.1.0 alpha.** The core is functional and tested; performance work
> and extra backends are in progress. This library is research software — it does
> not claim to beat state-of-the-art CNNs, but it is a clean, reproducible
> implementation of a legitimate reservoir-computing idea, and it is honest about
> what works and what doesn't.

---

## What it is

Reservoir computing separates a dynamical system (the "reservoir") from a simple
trainable readout. Only the readout is trained; the reservoir is fixed. If the
reservoir has rich enough internal dynamics, a linear readout can solve
surprisingly hard tasks.

`pixelflow` uses a 2D GPU texture as the reservoir state. Each pixel is a unit;
neighbours interact under a local rule implemented as a fragment shader (or an
equivalent numpy kernel on CPU). After `steps` iterations, the final texture is
flattened into a feature vector, and an `sklearn` linear model (Ridge, Logistic)
is trained on those features.

Three built-in rules:

| Rule | Dynamics | Use case |
|---|---|---|
| `diffusion_reaction` | Gray-Scott reaction-diffusion | Rich pattern formation — image features |
| `life_like` | Continuous Conway-style majority rule | Binary-ish inputs, texture classification |
| `wave` | Discrete wave equation (amplitude + velocity) | Eikonal / path-planning tasks |

## Install

```bash
pip install pixelflow              # core + CPU backend
pip install pixelflow[gpu]         # adds moderngl + glfw (OpenGL 3.3 GPU backend)
pip install pixelflow[cuda]        # adds cupy-cuda12x (CUDA backend)
pip install pixelflow[datasets]    # adds torchvision / pillow for dataset loaders
pip install pixelflow[all]         # everything
```

## Quickstart

```python
import numpy as np
from pixelflow import Reservoir, ReservoirConfig, RidgeReadout
from pixelflow.tasks.synthetic import two_moons

X_train, y_train = two_moons(n=500, noise=0.2, seed=0)
X_test,  y_test  = two_moons(n=200, noise=0.2, seed=1)

cfg = ReservoirConfig(
    width=32, height=32, channels=4,
    steps=8, rule="diffusion_reaction",
    input_encoding="project", seed=0,
)
res = Reservoir(cfg, backend="cpu")          # or backend="moderngl"

F_train = res.transform(X_train)             # (500, 32*32*4)
F_test  = res.transform(X_test)

readout = RidgeReadout(alpha=1.0).fit(F_train, y_train)
print(f"accuracy: {readout.score(F_test, y_test):.3f}")
```

Run the bundled example:

```bash
python examples/quickstart.py
```

## Backends

- **`cpu`** — pure NumPy. Always available. Reference implementation.
- **`moderngl`** — headless OpenGL 3.3 core via moderngl. Requires a GPU with
  working OpenGL drivers. Install with `pip install pixelflow[gpu]`.
- **`cuda`** — CuPy-backed CUDA implementation. Requires an NVIDIA GPU and
  matching CUDA toolkit (12.x or 13.x). Install with
  `pip install pixelflow[cuda]`.

All three backends produce numerically equivalent outputs (CPU vs moderngl:
max abs diff < 1e-5; CPU vs CUDA: max abs diff < 1e-3), verified in
`tests/test_moderngl_backend.py` and `tests/test_cuda_backend.py`.

## Benchmarks

Real measured results on MNIST (60k train / 10k test, `cpu` backend, Windows
10 + Python 3.13, raw JSON under `benchmarks/results/`):

| Rule | Grid | Steps | Test acc. | Raw-pixel baseline |
|---|---|---|---|---|
| `wave` | 32×32×4 | 4 | **0.9281** | 0.9261 |
| `diffusion_reaction` | 32×32×4 | 8 | 0.9213 | 0.9261 |

The `wave` reservoir slightly beats the raw-pixel LogReg baseline at these
first-order settings; no hyperparameter search was run. All numbers are
measured, not claimed. Losses are reported as-is.

## Why bother?

Reservoir computing has three practical strengths:

1. **Training is cheap** — only the readout is trained (closed-form for Ridge).
2. **The reservoir is reusable** — same features serve multiple downstream tasks.
3. **Non-standard hardware is plausible** — any substrate with rich local
   dynamics (optical systems, memristor arrays, analog chips) can act as a
   reservoir. GPU textures are a software twin of that.

`pixelflow` is not trying to beat ResNet on ImageNet. It's a clean
testbed for reservoir-computing research that happens to run fast on commodity
GPUs, and a pedagogical artifact for the "rendering is thinking" intuition.

## Project layout

```
pixelflow/
├── pixelflow/
│   ├── core/          # Reservoir, ReservoirConfig, CA rules, encoders
│   ├── backends/      # cpu (numpy), moderngl (GPU)
│   ├── readouts/      # Ridge, Logistic (sklearn wrappers)
│   └── tasks/         # MNIST, Eikonal, synthetic
├── tests/             # pytest suite (CPU-only tests always run)
├── benchmarks/        # honest measurements with JSON outputs
├── examples/          # runnable demos
├── paper/             # draft paper + figures
└── docs/              # API contract, design notes
```

## Citation

If you use `pixelflow` in research, please cite:

```bibtex
@software{angulo_pixelflow_2026,
  author  = {Angulo de Lafuente, Francisco},
  title   = {pixelflow: GPU Texture Reservoir Computing},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Agnuxo1/pixelflow},
  license = {Apache-2.0}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).

## Author

**Francisco Angulo de Lafuente**
GitHub: [@Agnuxo1](https://github.com/Agnuxo1)

## Acknowledgements

`pixelflow` distills a single clean idea from several years of experiments
across the NEBULA / NeuroCHIMERA / RED_NEURONAL_ANALOGICA project family.
The core reservoir-computing concept builds on classical work by Jaeger (Echo
State Networks) and Maass (Liquid State Machines); the GPU cellular-automaton
substrate is inspired by physical reservoir computing literature (Tanaka et
al. 2019) and practical fragment-shader-as-compute patterns.
