# ChurnBench Paper

**Target venue:** IEEE Access (open-access journal)

**Working title:** *ChurnBench: A Drift-Aware Benchmark for Grounding Agentic AI
over Enterprise Data Fabrics*

---

## Placeholder Convention

Two placeholder strings appear throughout `churnbench-paper.tex` and must
**never** be filled with invented or estimated figures:

| Placeholder | Meaning |
|---|---|
| `[DATA REQUIRED]` | A number, table cell, or prose claim that must come from a committed experiment artifact in `results/*.json`. Fill only after the corresponding experiment is run and its output file is committed. |
| `[CITATION NEEDED]` | A claim that needs a real reference. Add the entry to `references.bib` and replace the marker simultaneously. |
| `[FIGURE PENDING]` | A figure that must exist in `paper/figures/` before submission. |

The rule exists so that every number in the final submission is traceable to a
specific committed artifact — not to a draft note or memory.

---

## Paper–Code Relationship

This repository is the **reference implementation and benchmark** for the paper.

- `churnbench/generator/` — the timeline simulator described in §3
- `churnbench/ledger/` — the ground-truth ledger described in §3.2
- `churnbench/fabric/projector.py` — the frozen-at-$T$ projector (§3.4)
- `churnbench/tasks/` — task generator producing the §5 task suite *(in progress)*
- `churnbench/arms/` — agent arm implementations evaluated in §6 *(in progress)*
- `churnbench/eval/` — scoring; computes freshness accuracy (§4) *(in progress)*
- `results/` — committed experiment summaries that fill `[DATA REQUIRED]` slots

---

## Build Instructions

### Prerequisites

1. Install `latexmk` and a TeX distribution (e.g. MacTeX / TeX Live).

2. Copy `ieeeaccess.cls` from the [IEEE Access Author Kit](https://ieeeaccess.ieee.org/guide-for-authors/submit-your-article/)
   into the `paper/` directory. **Do not commit it** — it is not freely
   redistributable and is listed in `.gitignore`.

   ```sh
   cp /path/to/author-kit/ieeeaccess.cls paper/
   ```

3. `IEEEtran.bst` is already committed (LPPL license).

### Compile

```sh
cd paper
latexmk -pdf churnbench-paper.tex
```

The output PDF is `paper/churnbench-paper.pdf` (gitignored).

### Clean

```sh
cd paper
latexmk -C
```

---

## Results → Paper Traceability

Experiment scripts must write results to `results/<experiment>.json`.
Raw result files are gitignored; committed summaries (prefixed `summary_`)
are the authoritative source for filling `[DATA REQUIRED]` placeholders.

```
results/
  .gitkeep
  *.json          ← gitignored (raw experiment output)
  summary_*.json  ← committed (curated artifact, fills placeholders)
```
