# ChurnBench

**A drift-aware benchmark for grounding agentic AI over enterprise data fabrics.**

ChurnBench is an open-source evaluation instrument that measures how agentic systems
handle **temporal drift** across a heterogeneous enterprise stack: a SQL warehouse,
an operational NoSQL store, a rate-limited SaaS-style REST API, and an unstructured
document corpus.

Every mutation in the simulated fabric is written to a ground-truth ledger, so an
agent's answer at time `T` can be scored against the true world state at `T` —
making **freshness error** measurable, not merely qualitative.

Companion artifact for the paper:
> *Grounded and Current: A Grounding Architecture and Drift-Aware Benchmark
> for Agentic AI over Enterprise Data Fabrics*
> — IEEE Access (in preparation). See [`paper/`](paper/).

## Domain
Software Asset Management (SAM): licenses, users, consumption events, cost centers,
contracts. Drift is intrinsic — licenses get reassigned, contracts renew, users
offboard, prices change.

## Stack
- Postgres 16 — historical/analytical warehouse (star-schema-ish)
- MongoDB 7 — current operational state
- FastAPI — mock SaaS API (rate-limited, intentionally slow)
- Local document store — synthetic contracts and policies (markdown/PDF)
- Everything runs with `docker compose up`

## Quickstart
```bash
# Bring up the fabric
docker compose -f infra/docker-compose.yml up -d

# Install
poetry install

# Generate a dataset at a chosen timeline length
poetry run churnbench generate --days 180 --seed 42 --out ./data/run_001

# Snapshot the world at timestamp T
poetry run churnbench snapshot ./data/run_001 2024-09-15

# Generate evaluation tasks (coming soon)
poetry run churnbench tasks ./data/run_001 --n 180

# Run an experimental arm (coming soon)
poetry run churnbench run framework ./data/run_001

# Score results (coming soon)
poetry run churnbench score ./data/run_001
```

## Experimental arms
1. `naive` — agent hits raw sources live per query, no pre-processing
2. `classic-rag` — everything dumped in one vector store, uniform chunking
3. `framework` — the proposed grounding architecture (semantic model, source-aware
   routing, freshness-tiered caching)
4. `framework-no-cache` / `framework-no-semantic` — ablations

## Metrics
- Accuracy vs. ledger ground truth
- Cost per task (USD)
- End-to-end latency
- **Freshness error** — answer correct at T′ but wrong at T

## Paper

The [`paper/`](paper/) directory contains the IEEE Access draft LaTeX source
(`churnbench-paper.tex`).

**This repository is the reference implementation and benchmark for the paper.**
The code in `churnbench/` is the artifact described in the paper; results produced
by running the benchmark fill the `[DATA REQUIRED]` placeholders in the LaTeX
source.  See [`paper/README.md`](paper/README.md) for build instructions and the
placeholder convention.

Experiment scripts write outputs to `results/*.json` (gitignored raw) and
`results/summary_*.json` (committed). A `[DATA REQUIRED]` marker is only replaced
when a corresponding `summary_*.json` is committed — never with an estimated number.

## License
MIT (planned)

## Citation
BibTeX entry will be added once the paper is published.
