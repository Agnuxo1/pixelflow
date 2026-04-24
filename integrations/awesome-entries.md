# Awesome-list entries (draft)

Exact markdown lines to paste when submitting PRs. One entry per list, short,
honest, pointing to measured benchmarks.

---

## awesome-reservoir-computing

Target repo: <https://github.com/reservoirpy/awesome-reservoir-computing>
(or `reservoirpy/awesome-reservoir-computing` — verify the canonical one
before PR)

Section: **Software libraries** (alphabetical)

```markdown
- [pixelflow](https://github.com/Agnuxo1/pixelflow) — GPU Texture Reservoir
  Computing. A 2D RGBA texture evolves under local cellular-automaton rules
  (Gray–Scott, continuous Life, discrete wave) expressed as fragment shaders;
  a linear readout (Ridge / Logistic) is trained on the flattened final
  state. Three backends (NumPy / moderngl / CUDA), batched CUDA path gives
  ~18× speedup over CPU on an RTX 3090. Apache-2.0.
```

## awesome-local-ai

Target repo: <https://github.com/janhq/awesome-local-ai> (or community fork)

Section: **Libraries & Frameworks** (or nearest)

```markdown
- [pixelflow](https://github.com/Agnuxo1/pixelflow) — Tiny research library
  that turns a GPU texture into a reservoir-computing substrate for image
  classification and simple PDE tasks. Runs fully local on CPU, OpenGL, or
  CUDA; no network, no telemetry. Apache-2.0.
```

## awesome-neuromorphic-computing

Target repo: <https://github.com/open-neuromorphic/open-neuromorphic>
or similar curated list (verify before PR).

Section: **Reservoir / analog compute**

```markdown
- [pixelflow](https://github.com/Agnuxo1/pixelflow) — Software twin of a
  physical reservoir: pixels on a GPU texture act as coupled units evolving
  under local CA rules, a linear readout is trained on top. Positioned as a
  testbed for the class of analog/optical reservoirs discussed in
  Tanaka et al. 2019.
```

---

## Submission protocol

For each list:

1. Verify the list is actively maintained (last commit < 6 months).
2. Read its CONTRIBUTING.md; match alphabetical / categorical ordering exactly.
3. Fork, branch `add-pixelflow`, single-commit PR with the entry above.
4. PR title: `Add pixelflow — GPU Texture Reservoir Computing`.
5. PR body: 2–3 sentences + link to v0.3.0 release + link to benchmarks JSON.
6. If rejected / ignored for > 4 weeks: do NOT bump, do NOT re-open. Accept.
