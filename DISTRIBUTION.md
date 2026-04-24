# pixelflow — distribution log

Honest record of where pixelflow has been submitted / listed / integrated, with
status and links. No aspirational entries — only things actually done.

## Published

| Channel | URL | Status | Date |
|---|---|---|---|
| GitHub repo | <https://github.com/Agnuxo1/pixelflow> | Live, v0.3.0 released | 2026-04-24 |
| GitHub release v0.3.0 | <https://github.com/Agnuxo1/pixelflow/releases/tag/v0.3.0> | Live | 2026-04-24 |

## Prepared, awaiting credentials

| Channel | Notes |
|---|---|
| PyPI (`pixelflow-rc`) | Dist built (`dist/pixelflow_rc-0.3.0*`), passes `twine check`. Name `pixelflow` on PyPI is taken by an unrelated CV library, so the dist name is `pixelflow-rc` (import remains `import pixelflow`). Upload blocked: provided API token returns `403 Invalid API Token: current user does not match user restriction in token`. Needs a fresh token from the PyPI account owner. |
| HuggingFace Space | Code in `space/` is complete (Gradio 4.x, 2 tabs). `huggingface-cli` not authenticated in this environment. |

## In preparation

| Channel | Artifact | Notes |
|---|---|---|
| LlamaIndex adapter | `integrations/llamahub/` | End-to-end smoke test passing against `llama-index-core` 0.12+. Standalone class, no upstream PR needed — can be distributed via PyPI `pixelflow-rc` directly. |
| Open WebUI tool | `integrations/open-webui/` | End-to-end smoke test passing. Upload to openwebui.com via the tool-creator UI — no automated API, manual step for user. |

## Investigated and deliberately skipped

| Channel | Reason |
|---|---|
| `reservoirpy/awesome-reservoir-computing` | Last push 2021-11, effectively abandoned. |
| `janhq/awesome-local-ai` | LLM-centric; pixelflow is not an LLM tool. Submitting would be spam. |
| `open-neuromorphic/open-neuromorphic` | Curated SNN list with a strict table/image format; pixelflow is reservoir computing, not SNN — off-topic. |
| LangChain / CrewAI / Haystack wrappers | Trivial wrapper with no real value beyond the LlamaIndex adapter. Would pad the project without substance. |
| LlamaHub PR to `run-llama/llama_index` | `BaseLlamaPack` removed in 0.12 restructure; packs as a concept are deprecated upstream. Proper `llama-index-integrations` PR is a multi-day effort with CI constraints — not worth half-baking. |
| VS Code / JetBrains / Chrome extensions | No meaningful UI surface for a numeric research library. Won't submit placeholder extensions. |

## Deferred (require user credentials or review)

- Chrome Web Store, VS Code Marketplace, JetBrains Marketplace — no natural
  browser/IDE surface for a numeric reservoir library yet. Will add only when
  a real integration exists, not to pad the list.
- Play Store / App Store — out of scope for v0.x. A mobile demo would need
  substantial extra work; honest to defer.
- arXiv — explicit user instruction: user submits personally. Bundle in
  `paper/arxiv/`.

## Principles

- No spam. One well-targeted entry per list, with working install + link to
  measured benchmarks.
- No fake stars, no sockpuppets, no review theater.
- Every entry links back to a reproducible command and a JSON artifact under
  `benchmarks/results/`.
