# Submitting pixelflow to Papers With Code

Papers With Code (paperswithcode.com) indexes ML papers + code with benchmark
leaderboards. Submission is manual (no API); these are the exact steps.

## Prerequisites

- GitHub account (can link via OAuth)
- The arXiv preprint URL (once you submit; see `paper/arxiv/`)

## Steps

### 1. Add a paper
Go to: <https://paperswithcode.com/add-paper>

Fill in:
- **Paper URL**: your arXiv URL, e.g. `https://arxiv.org/abs/XXXX.XXXXX`
- **GitHub URL**: `https://github.com/Agnuxo1/pixelflow`
- Click *Submit*

If the paper is not yet on arXiv you can submit the GitHub repo directly:
<https://paperswithcode.com/add-repo> → paste `Agnuxo1/pixelflow`.

### 2. Link to the MNIST benchmark

After the paper/repo appears:
- Go to the paper page → *Edit* → *Benchmarks*
- Dataset: **MNIST**
- Task: **Image Classification**
- Metric: **Accuracy**
- Add row: method `pixelflow (wave, 32×32×4, 4 steps, CUDA)`, value `0.9281`

### 3. Link to CIFAR-10 (honest negative)

Same procedure, CIFAR-10 dataset, acc `0.2430` (grayscale, 10k subset).
Papers With Code does not penalise results below baseline — transparency is
encouraged.

## Result

Your paper gets a "code" badge on arXiv and appears in reservoir-computing
and cellular-automata method searches on PwC.
