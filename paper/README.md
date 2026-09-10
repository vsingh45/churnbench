# ChurnBench Paper

> **The authoritative build lives outside this repository.**
> `paper/churnbench-aixse.pdf` — the actual submission draft, currently
> targeting AIxSE and used for the arXiv cs.SE endorsement request — is
> produced from a source that is **not** `churnbench-paper.tex` below.
> That `.tex` file predates the paper's current title, results, and
> section structure, and still carries `[DATA REQUIRED]` placeholders
> throughout. Do not edit it expecting it to affect the submitted PDF,
> and do not treat it as the paper's current text — it isn't.
>
> To correct the submitted PDF: fix it at its actual source (external to
> this repo), then replace `paper/churnbench-aixse.pdf` here and cut a new
> release. This repo's job is to host and version the PDF and the data
> that backs its numbers ([REVIEWERS.md](../REVIEWERS.md)), not to build it.

**Target venue:** IEEE Access (open-access journal) — *superseded; current
target is AIxSE, see the submission PDF itself for the live title and venue.*

**Title (this `.tex` file only, stale):** *Grounded and Current: A Grounding
Architecture and Drift-Aware Benchmark for Agentic AI over Enterprise Data
Fabrics*

**Status:** This `.tex` source is an abandoned early draft — all experiment
sections still contain `[DATA REQUIRED: …]` placeholders below. It is kept
for the build-instructions and placeholder-convention reference only; it is
not being actively edited toward the current submission.

---

## Placeholder Convention

Three marker strings appear throughout `churnbench-paper.tex`:

| Marker | Meaning | Rule |
|---|---|---|
| `[DATA REQUIRED: …]` | A number, table cell, figure, or prose claim that must come from a committed experiment artifact in `results/summary_*.json`. | Never fill with an estimated or remembered value. Run the experiment, commit the summary, then fill. |
| `[CITATION NEEDED: …]` | A claim that needs a verified reference. | Add the entry to the inline `thebibliography` block and replace the marker in the same commit. |
| `[… TODO]` | A structural placeholder (author list, URL, venue status). | Resolve before submission; not gated on experiments. |

---

## Paper–Code Relationship

This repository is the **reference implementation and benchmark** for the paper.
Every code module maps to a paper section:

| Module | Paper section |
|---|---|
| `churnbench/generator/timeline.py` | §4.2 Timeline Generation |
| `churnbench/ledger/ledger.py` | §4.2 Ground-Truth Ledger |
| `churnbench/fabric/projector.py` | §4.3 Frozen-at-T Protocol |
| `churnbench/tasks/` | §4.4 Task Generation *(in progress)* |
| `churnbench/arms/` | §5 Experimental Arms *(in progress)* |
| `churnbench/eval/` | §4.5 Metrics / §6 Results *(in progress)* |
| `results/summary_*.json` | Fills `[DATA REQUIRED]` slots in §6 |

The paper's §3 (Grounding Architecture) describes the design; the code in
`churnbench/` is the artifact that instantiates it for evaluation.

---

## Build Instructions

### Prerequisites

1. **TeX distribution** — MacTeX, TeX Live, or MiKTeX with `latexmk`.

2. **`ieeeaccess.cls`** — copy from the
   [IEEE Access Author Kit](https://ieeeaccess.ieee.org/guide-for-authors/submit-your-article/)
   into `paper/`. **Do not commit it** — it is not freely redistributable
   and is listed in the root `.gitignore`.

   ```sh
   cp /path/to/author-kit/ieeeaccess.cls paper/
   ```

3. **`IEEEtran.bst`** — already committed (LPPL license). Not used by the
   current draft (inline `thebibliography`) but kept for a future BibTeX
   migration.

### Compile

```sh
cd paper
latexmk -pdf churnbench-paper.tex
```

Output: `paper/churnbench-paper.pdf` (gitignored).

### Clean

```sh
cd paper
latexmk -C
```

---

## Results → Paper Traceability

All experiment scripts must write outputs to `results/`. Raw files are gitignored;
only committed summaries fill `[DATA REQUIRED]` slots.

```
results/
  .gitkeep
  *.json           ← gitignored — raw experiment output
  summary_*.json   ← committed — curated artifact; the only valid source
                     for filling [DATA REQUIRED] placeholders
```

A `[DATA REQUIRED]` marker is replaced in `churnbench-paper.tex` only when the
corresponding `summary_*.json` is committed in the **same** or **prior** commit.
This makes every number in the final PDF traceable to `git log`.

---

## Open TODOs Before Submission

- [ ] Confirm co-author list (`[AUTHOR TODO]`)
- [ ] Resolve venue/status for `\cite{par}`, `\cite{saasrat}`, `\cite{gte}`
- [ ] Fill `[AUTHORS TODO]` entries in `thebibliography`
- [ ] Add MS MARCO, MuSiQue, 2WikiMultihopQA `[CITATION NEEDED]` entries
- [ ] Add canonical data-fabric / semantic-layer references (`[CITATION NEEDED]`)
- [ ] Anchor Poisson drift means to published churn statistics (`[CITATION NEEDED]`)
- [ ] Add Onyx URL (`[URL TODO]`)
- [ ] Produce `fig:arch` architecture diagram → `paper/figures/fig_arch.pdf`
- [ ] Fill all `[DATA REQUIRED]` blocks from committed `results/summary_*.json`
