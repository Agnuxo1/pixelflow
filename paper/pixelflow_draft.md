# pixelflow: GPU Texture Reservoir Computing

**Francisco Angulo de Lafuente**
Draft v0.1 · 2026

## Abstract

We present `pixelflow`, an open-source library for reservoir computing on GPU
textures. A 2D RGBA texture evolves under a local cellular-automaton rule
expressed as a fragment shader; a linear model is trained on the final texture
state to solve downstream tasks. The approach is a software analogue of
physical reservoir computing and exploits the GPU's native strength — massively
parallel per-pixel updates with local neighbourhood access. We demonstrate the
library on image classification (MNIST, two-moons) and PDE approximation
(Eikonal equation). The implementation provides a CPU NumPy reference backend
and a headless OpenGL (moderngl) GPU backend that agree numerically to within
1e-3. The library is released under Apache-2.0 at
<https://github.com/Agnuxo1/pixelflow>.

## 1. Introduction

Reservoir computing (RC) is a training paradigm in which a fixed, high-dimensional
dynamical system is used to project inputs into a rich feature space, and only a
linear readout is trained [Jaeger 2001, Maass 2002]. RC has been instantiated in
software (echo state networks, liquid state machines) and in physical substrates
(optical, memristor, spintronic) [Tanaka et al. 2019].

This work treats a GPU texture as the reservoir substrate. Each pixel holds a
4-channel state (RGBA); a fragment shader applies a local update rule using the
standard texture-sampling primitives GPUs are optimized for. After `k` update
steps, the flattened texture becomes the feature vector for a scikit-learn
linear readout.

## 2. Method

### 2.1 State and dynamics

Let `S_t ∈ R^{H×W×4}` be the reservoir state at step `t`. The initial state
`S_0` is derived from an input `x ∈ R^d` by one of three encoders:

- **tile**: x is reshaped/resized to (H, W) and placed in channel 0.
- **phase**: scalar x_i are mapped to (sin, cos) in channels 0, 1.
- **project**: fixed random Gaussian matrix W ∈ R^{(H·W·4)×d}, seed-driven.

The update rule `S_{t+1} = Φ(S_t)` is one of three families:

- **diffusion_reaction** (Gray-Scott): two-channel reaction-diffusion with
  feed rate F and kill rate k.
- **life_like**: continuous relaxation of Conway's Life on channel 0.
- **wave**: discrete wave equation with amplitude + velocity channels.

All rules use periodic boundary conditions implemented as a wrap sampler.

### 2.2 Readout

Features `φ(x) = flatten(S_k)` are fed to a `sklearn` linear model:
`RidgeClassifier` or `LogisticRegression`. Training is closed-form for Ridge
and convex for Logistic — no backprop through the reservoir.

### 2.3 Backends

Two backends implement `Φ` identically up to floating-point tolerance:

- **cpu**: vectorized NumPy with `np.roll` for periodic neighbourhood access.
- **moderngl**: headless OpenGL 3.3 core, ping-pong framebuffers, GL_RGBA32F
  textures, fragment shader per rule. Standalone context creation via
  `moderngl.create_standalone_context()`.

CPU/GPU parity is verified by `tests/test_moderngl_backend.py`.

## 3. Experiments

*(To be populated from `benchmarks/results/` once runs complete.)*

| Task | Reservoir | Readout | Accuracy | Baseline |
|---|---|---|---|---|
| two-moons (n=500) | 32×32×4, 8 steps, diffusion | Ridge | TBD | Logistic on raw: TBD |
| MNIST (subset=10k) | 64×64×4, 16 steps, life_like | Logistic | TBD | Logistic on raw: TBD |
| Eikonal (constant-speed) | 128×128×4, 64 steps, wave | Ridge | TBD | scipy fast-marching |

## 4. Honest limitations

- The library does not aim to match deep CNNs on image classification. For
  MNIST, a simple CNN reaches >99%; `pixelflow`'s goal is clean reservoir
  dynamics, not benchmark chasing.
- For small batches (<1000 samples) on small grids (<32×32), the GPU backend
  is often slower than CPU due to context overhead. Benefit emerges at scale.
- The CA rules are simple choices; learned reservoir dynamics
  (meta-learned CA à la Mordvintsev et al.) are future work.

## 5. Reproducibility

All benchmarks are runnable as `python benchmarks/<name>.py` and emit JSON to
`benchmarks/results/`. Seeds are fixed in configs. Hardware used for any
reported numbers is declared in the JSON metadata.

## References

- Jaeger, H. (2001). *The "echo state" approach to analysing and training
  recurrent neural networks.* GMD Report 148.
- Maass, W., Natschläger, T., & Markram, H. (2002). *Real-time computing
  without stable states: A new framework for neural computation.* Neural
  Computation 14(11).
- Tanaka, G. et al. (2019). *Recent advances in physical reservoir computing:
  A review.* Neural Networks 115.
- Mordvintsev, A. et al. (2020). *Growing neural cellular automata.* Distill.

## License and citation

Apache-2.0. Cite via the BibTeX entry in the project README.
