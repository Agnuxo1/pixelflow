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
| awesome-reservoir-computing | Entry drafted in `integrations/awesome-entries.md` | Needs PR to the list repo |
| awesome-local-ai | Entry drafted | Needs PR |
| LlamaIndex LlamaHub | Pack under `integrations/llamahub/` | Needs PR to `run-llama/llama_index` |
| Open WebUI tools | Tool under `integrations/open-webui/` | Needs listing on openwebui.com |

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
