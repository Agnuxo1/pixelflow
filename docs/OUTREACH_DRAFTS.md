# Outreach drafts — pixelflow

Ready-to-post text for the highest-impact channels. Francisco posts from his
own accounts. No bulk posting, no cross-posting to irrelevant places.

---

## 1. Hacker News — "Show HN" (highest ROI, ideal for pixelflow)

**Title:**
```
Show HN: pixelflow – GPU textures as reservoir computers (17× faster than NumPy on RTX 3090)
```

**Body (plain text, no markdown on HN):**
```
pixelflow treats a 2-D RGBA GPU texture as a fixed random dynamical system —
a "reservoir" in the reservoir-computing sense (Jaeger 2001, Maass 2002). Each
pixel interacts with its neighbours under a local rule implemented as a GLSL
fragment shader (Gray-Scott reaction-diffusion, a discrete wave equation, or a
continuous Conway-majority rule). After N steps the final texture is flattened
and a ridge or logistic regression is trained on top. Only the linear readout
is ever trained; the reservoir is fixed.

Three backends: pure NumPy (always available), moderngl (headless OpenGL 3.3),
and CuPy CUDA. The CUDA backend processes all N samples as a single 4-D tensor
(N, H, W, C), transferring to and from the GPU exactly once. On an RTX 3090:

  N=1000, 64×64, 32 steps, diffusion-reaction:
  CPU 5.00 s → CUDA batched 0.28 s (17.95×)

On MNIST 60k (not a subset): wave rule, 32×32×4, 4 steps, logistic readout:
  accuracy 0.9281 vs. raw-pixel logistic baseline 0.9261.

On CIFAR-10 (grayscale, 10k subset): 0.2430 vs 0.2521 baseline —
honest negative, documented in the README.

Repo: https://github.com/Agnuxo1/pixelflow
Demo: https://huggingface.co/spaces/Agnuxo/pixelflow
Install: pip install pixelflow-rc

The interesting question isn't whether it beats ResNet — it doesn't. It's
whether a fixed, non-learned substrate with rich dynamics can close most of
the gap with a tiny linear readout, and what that says about the information
content already present in the physics of the rule.
```

**When to post:** Tuesday or Wednesday, 9-11 AM US Eastern.

---

## 2. Reddit r/MachineLearning — research post

**Title:**
```
[Project] pixelflow: GPU texture as a reservoir computer — 17× CUDA speedup, honest MNIST/CIFAR benchmarks, Apache-2.0
```

**Body:**
```
Reservoir computing (Jaeger's Echo State Networks, Maass's Liquid State Machines)
fixes a high-dimensional dynamical system and trains only a linear readout on top.
pixelflow implements this with a 2-D GPU texture as the reservoir:

- Each pixel = one unit; neighbours interact under a local CA rule (GLSL shader
  or NumPy equivalent for CPU)
- Three rules: Gray-Scott reaction-diffusion, discrete wave equation, continuous
  Conway-majority
- Three backends: NumPy, OpenGL (moderngl), batched CUDA (cupy)
- The CUDA path processes a full batch of N samples as a (N,H,W,C) tensor with
  a single host↔device transfer — 17.95× over NumPy on RTX 3090

Honest numbers on MNIST 60k:
  wave rule, 32×32×4, 4 steps → acc 0.9281 vs. 0.9261 raw-pixel logistic baseline
  CUDA transform time: 5.81 s (vs. 24.15 s on CPU)

Honest negative on CIFAR-10 (grayscale 10k subset):
  0.2430 vs. 0.2521 baseline — reservoir slightly underperforms; reported as-is.

The motivation isn't to beat CNNs. It's a clean software twin of physical
reservoir computers (optical systems, memristor arrays) where the substrate
physics is the computation, not the weights.

GitHub: https://github.com/Agnuxo1/pixelflow
PyPI: pip install pixelflow-rc
Demo: https://huggingface.co/spaces/Agnuxo/pixelflow
```

---

## 3. Twitter / X thread (7 tweets)

```
1/7
I built pixelflow: a GPU texture as a reservoir computer.

The idea: fix a 2-D grid of pixels evolving under a cellular-automaton rule.
Train only a linear layer on the frozen final state. No backprop through the
substrate.

github.com/Agnuxo1/pixelflow

2/7
Three built-in CA rules:
• Gray–Scott reaction-diffusion (rich pattern formation)
• Discrete wave equation (propagates signal, useful for path planning)
• Continuous Conway-majority (binary-ish dynamics)

Each implemented as both a GLSL fragment shader and a NumPy kernel.

3/7
Three backends:
• CPU: pure NumPy, always available
• moderngl: headless OpenGL 3.3 (any GPU with drivers)
• CUDA: cupy, batched — all N samples as one (N,H,W,C) tensor

The batched CUDA path does one host↔device transfer for the whole dataset.

4/7
On an RTX 3090:

N=1000, 64×64, 32 steps, Gray-Scott rule:
CPU ████████████████ 5.00 s
CUDA ▌ 0.28 s → 17.95× faster

N=2000, 32×32, 16 steps, wave rule:
CPU ████████ 2.13 s
CUDA ▌ 0.17 s → 12.32× faster

5/7
Honest MNIST numbers (60k train / 10k test):

wave rule, 32×32×4, 4 steps:
→ accuracy 0.9281
→ raw-pixel logistic baseline: 0.9261
→ reservoir beats baseline ✓

CIFAR-10 (grayscale, 10k subset):
→ reservoir 0.2430, baseline 0.2521
→ honest negative — documented, not hidden ✗

6/7
Why bother?

1. Training is cheap — only a ridge regression
2. The reservoir is reusable — same features, multiple tasks
3. Non-standard substrates (optical, memristors) can be reservoirs too.
   GPU textures are a software twin of that.

"Rendering is thinking."

7/7
pip install pixelflow-rc

Demo: huggingface.co/spaces/Agnuxo/pixelflow
Paper bundle: github.com/Agnuxo1/pixelflow/tree/main/paper/arxiv

Apache-2.0. All benchmarks are measured; raw JSON under benchmarks/results/.
```

---

## 4. LinkedIn (professional, shorter)

```
After months of research, I've published pixelflow — an open-source Python
library for GPU Texture Reservoir Computing.

The core idea: treat a 2-D GPU texture as a fixed dynamical system (cellular
automaton), train only a linear readout on top. No backpropagation through the
substrate. Three backends: NumPy, OpenGL, CUDA.

Key results on RTX 3090:
→ 17.95× speedup vs. CPU numpy (batched CUDA, 1000 samples)
→ MNIST 60k accuracy: 0.9281 (wave rule, 4 steps)
→ Honest negative on CIFAR-10 documented and not hidden

Everything is Apache-2.0, benchmarks are reproducible (raw JSON in the repo),
and the interactive demo runs on HuggingFace Spaces.

🔗 GitHub: github.com/Agnuxo1/pixelflow
📦 PyPI: pip install pixelflow-rc
🎮 Demo: huggingface.co/spaces/Agnuxo/pixelflow

#MachineLearning #ReservoirComputing #OpenSource #GPU #Python #NeuromorphicComputing
```

---

## Posting order (priority)

1. **Hacker News** — post once, no bumping. Peak time Tue/Wed 9-11 AM ET.
2. **r/MachineLearning** — wait 24h after HN.
3. **Twitter thread** — same day as HN, links back to HN thread for social proof.
4. **LinkedIn** — after Twitter.

## What NOT to do

- Do not post to r/learnmachinelearning, r/artificial, r/singularity — off-topic.
- Do not cross-post the same text to multiple subreddits simultaneously.
- Do not use bot accounts to upvote.
- Do not email ML newsletters unless they have a "submissions" page.
- Wait for organic engagement before posting in Discord servers.
