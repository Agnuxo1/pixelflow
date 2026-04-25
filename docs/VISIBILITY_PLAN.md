# Visibility plan — Francisco Angulo de Lafuente

Honest assessment + concrete action list. Separated into "done by Claude" and
"requires Francisco" so nothing is forgotten.

---

## Current situation (audited 2026-04-25)

**GitHub (Agnuxo1):**
- 97 repos total; ~55 are forked awesome-lists (noise, no original content)
- Top original repos by stars: OpenCLAW-P2P (35), The-Living-Agent (14),
  p2pclaw-mcp-server (14), ASIC-RAG-CHIMERA (4), pixelflow (1)
- Profile README: now live at github.com/Agnuxo1
- Homepage: pinning must be done via web UI (GitHub API does not expose it)

**HuggingFace (Agnuxo):**
- 53 spaces total; 4 RUNNING, ~12 SLEEPING (wake on traffic), ~37 PAUSED
- Running: pixelflow, OpenCLAW-Agent, p2pclaw-lean4-verifier, P2PCLAW-Benchmark
- Broken: p2pclaw-api (RUNTIME_ERROR), nebula-physicist-agent (NO_APP_FILE)
- No-SDK: evolutionary-intelligence-agi, climate-ai-agi (appear broken to visitors)

---

## Already done (no action needed)

| Task | Result |
|---|---|
| GitHub profile README | live: github.com/Agnuxo1 |
| pixelflow on PyPI | pip install pixelflow-rc 0.3.0 |
| pixelflow HF Space | running: huggingface.co/spaces/Agnuxo/pixelflow |
| Awesome-list PRs | artiomn/awesome-neuromorphic #6, ChristosChristofidis/awesome-deep-learning #304 |
| LlamaIndex adapter | integrations/llamahub/ |
| Open WebUI tool | integrations/open-webui/ |
| Papers With Code guide | docs/SUBMIT_PAPERWITHCODE.md |
| Outreach post drafts | docs/OUTREACH_DRAFTS.md |
| Repo cross-topics | ASIC-RAG-CHIMERA, QESN, The-Living-Agent updated |
| Repo homepages | pixelflow → HF Space, OpenCLAW-P2P → HF Space |

---

## Requires Francisco — ordered by impact

### 1. Pin repos on GitHub (5 minutes, web UI only)
GitHub API doesn't expose repo pinning. Do it at:
https://github.com/Agnuxo1 → "Customize your pins"

Pin these 6 (in this order):
1. OpenCLAW-P2P
2. pixelflow
3. The-Living-Agent
4. ASIC-RAG-CHIMERA
5. enigmagent-mcp
6. p2pclaw-mcp-server

### 2. Post on Hacker News — "Show HN" for pixelflow
Exact title and text in `docs/OUTREACH_DRAFTS.md` section 1.
Post Tuesday or Wednesday, 9–11 AM US Eastern (best HN timing).
**Do not repost if it doesn't gain traction — one shot only.**

### 3. Post on Reddit r/MachineLearning
Text in `docs/OUTREACH_DRAFTS.md` section 2.
Wait at least 24h after the HN post.
Subreddit rules: must be original research, no promotional language.
Check current subreddit rules at reddit.com/r/MachineLearning/wiki/posting_policy

### 4. Twitter / X thread (7 tweets)
Text in `docs/OUTREACH_DRAFTS.md` section 3.
Tag: @huggingface and @GradioML in tweet 7 — they sometimes RT community projects.

### 5. LinkedIn post
Text in `docs/OUTREACH_DRAFTS.md` section 4.
Shortest and safest — good for professional network.

### 6. Submit to Papers With Code
Instructions in `docs/SUBMIT_PAPERWITHCODE.md`.
Requires arXiv preprint URL first (you handle arXiv submission personally).

### 7. Fix broken HF Spaces (optional cleanup)
- `Agnuxo/evolutionary-intelligence-agi` — set sdk in README.md or delete it
- `Agnuxo/climate-ai-agi` — same
- `Agnuxo/nebula-physicist-agent` — add an app.py or delete
- `Agnuxo/p2pclaw-api` — fix RUNTIME_ERROR or pause it manually

These don't hurt you actively but look unprofessional to visitors.

### 8. Revive P2PCLAW-Hive-Research-Network (high effort, high reward)
The HF audit rated this as the best-documented space. It's PAUSED.
Restarting it and linking it from OpenCLAW-P2P GitHub would tie the two
biggest projects together visually.

---

## What NOT to do

- Do NOT bulk-post pixelflow to r/learnmachinelearning, r/artificial, r/technology —
  off-topic, looks like spam.
- Do NOT use bot services to buy GitHub stars or HF likes.
- Do NOT submit the same post to multiple subreddits at the same time.
- Do NOT delete the forked awesome-list repos (they have no negative effect and
  deleting 55 repos is risky; just let them sit).
- Do NOT post on josephmisiti/awesome-machine-learning — maintainer requires
  email submission due to LLM-spam PRs (as of 2026).

---

## Realistic expectations

GitHub stars come slowly for research tools. The path is:
1. HN / Reddit post → initial spike (50-200 visitors in 24h)
2. Organic citations in other researchers' READMEs
3. Papers With Code listing → academic discovery
4. The awesome-list PRs → passive long-tail discovery

Reservoir computing is a small but real community. pixelflow is honest and
reproducible — that matters. One good HN "Show HN" with verified benchmarks
is worth more than 100 spam posts.
