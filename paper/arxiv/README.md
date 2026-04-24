# arXiv submission bundle

Files for submitting the `pixelflow` technical report to arXiv (cs.NE).
**Submission is performed by the author** — the library maintainer does not
auto-submit.

## Contents

- `pixelflow.tex` — main LaTeX source, self-contained.
- (optional) `figures/` — add any generated plots before submission.

## Build

```bash
cd paper/arxiv
pdflatex pixelflow.tex
bibtex pixelflow        # only if external .bib is added
pdflatex pixelflow.tex
pdflatex pixelflow.tex
```

Or simply upload `pixelflow.tex` to arXiv and let their compiler handle it.

## Submission checklist

1. Read current draft end-to-end; update version/date if >1 day old.
2. Add any missing experiment tables from `benchmarks/results/`.
3. Choose primary category: **cs.NE** (Neural and Evolutionary Computing).
   Cross-list: **cs.LG**, optionally **eess.IV**.
4. License: arXiv perpetual + code is Apache-2.0 (already stated).
5. Submit via <https://arxiv.org/submit>.

## After submission

- Add the arXiv badge to the main README.md.
- Update the citation BibTeX entry in README.md with the arXiv ID.
- Create a `CITATION.cff` at the repo root.
