---
title: pixelflow
emoji: 🌊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# pixelflow — GPU Texture Reservoir Computing

Live interactive demo of the `pixelflow` library, which treats image pixels on a
GPU texture (or CPU grid) as a fixed random reservoir evolving under local
cellular-automaton rules, and trains a lightweight linear readout on top for
classification tasks.

## How it works

Reservoir computing fixes the dynamical system (the "reservoir") and only trains
a simple linear layer (the "readout"). In `pixelflow` the reservoir is a 2-D grid
of pixels where each cell interacts with its neighbours under one of three built-in
cellular-automaton rules — `diffusion_reaction` (Gray-Scott), `life_like`
(continuous Conway majority), or `wave` (discrete wave equation). After a chosen
number of steps the final grid is flattened into a feature vector, and a logistic
or ridge regression is fitted on those features. This demo runs entirely on CPU;
GPU/OpenGL backends (via `moderngl`) are available locally but require a discrete
GPU and are not used here.

## Tabs

- **CA Visualizer** — choose a rule, grid size, and number of steps; see the
  initial random state and the evolved state side by side, plus a strip of eight
  intermediate frames.
- **MNIST Classifier Demo** — draw a digit on the sketchpad; the backend runs a
  pre-trained Reservoir + LogisticReadout pipeline trained at startup on 2000 MNIST
  samples and returns the predicted class with a probability bar chart and a
  heatmap of the reservoir's intermediate state.

## GitHub

<https://github.com/Agnuxo1/pixelflow>

## Citation

```bibtex
@software{angulo2024pixelflow,
  author  = {Angulo de Lafuente, Francisco},
  title   = {pixelflow: GPU Texture Reservoir Computing},
  year    = {2024},
  url     = {https://github.com/Agnuxo1/pixelflow},
  license = {Apache-2.0},
}
```
