# pixelflow — distribution log

Honest record of where pixelflow has been submitted / listed / integrated, with
status and links. No aspirational entries — only things actually done.

## Published

| Channel | URL | Status | Date |
|---|---|---|---|
| GitHub repo | <https://github.com/Agnuxo1/pixelflow> | Live, v0.3.0 | 2026-04-24 |
| GitHub release v0.3.0 | <https://github.com/Agnuxo1/pixelflow/releases/tag/v0.3.0> | Live | 2026-04-24 |
| PyPI (`pixelflow-rc`) | <https://pypi.org/project/pixelflow-rc/0.3.0/> | Live | 2026-04-25 |
| HuggingFace Space | <https://huggingface.co/spaces/Agnuxo/pixelflow> | Running (Gradio 5) | 2026-04-25 |

## Integrations (in repo, ready to use)

| Channel | Artifact | Notes |
|---|---|---|
| LlamaIndex adapter | `integrations/llamahub/` | Standalone `PixelflowReservoirPack`; distributed via `pixelflow-rc` on PyPI. |
| Open WebUI tool | `integrations/open-webui/pixelflow_tool.py` | Drop-in `Tools` class. Upload via openwebui.com UI (no automated API). |

## Investigated and deliberately skipped

| Channel | Reason |
|---|---|
| `reservoirpy/awesome-reservoir-computing` | Last push 2021-11, effectively abandoned. |
| `janhq/awesome-local-ai` | LLM-centric; pixelflow is not an LLM tool. Submitting would be spam. |
| `open-neuromorphic/open-neuromorphic` | Curated SNN list with strict format; reservoir computing ≠ SNN — off-topic. |
| LangChain / CrewAI / Haystack wrappers | No real value beyond the LlamaIndex adapter. |
| LlamaHub PR to `run-llama/llama_index` | `BaseLlamaPack` removed in 0.12; packs deprecated upstream. Multi-day CI effort — not worth half-baking. |
| VS Code / JetBrains / Chrome extensions | No meaningful UI surface for a numeric research library. |
| Play Store / App Store | Out of scope for v0.x. |

## Deferred (require user action)

- **arXiv** — explicit user instruction: user submits personally. Bundle ready in `paper/arxiv/`.
- **openwebui.com listing** — requires manual upload via the tool-creator web UI; no automated API.

## Principles

- No spam. One well-targeted entry per list, with working install + measured benchmarks.
- No fake stars, no sockpuppets, no review theater.
- Every entry links to a reproducible command and a JSON artifact under `benchmarks/results/`.
