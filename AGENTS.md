# ChurnBench — Agent Guide

Onboarding for coding agents (Claude Code and compatible) working in this repo.
ChurnBench is a **drift-aware benchmark** for grounding agentic AI over enterprise
data fabrics (Postgres warehouse + MongoDB operational store + rate-limited SaaS
REST API + document corpus). Every mutation is written to a ground-truth ledger so
an agent's answer at time `T` is scored against the true world state at `T`.

## Stack

- **Python 3.11**, packaged with **Poetry** (`pyproject.toml`).
- CLI via **Typer** (`churnbench` entry point → `churnbench.cli:app`).
- Fabric: **Postgres 16**, **MongoDB 7**, a **FastAPI** mock SaaS API — all via
  `docker compose` under `infra/`.
- Agentic layer: **LangGraph / LangChain** (+ `langchain-openai`, `langchain-anthropic`).
- RAG: **ChromaDB** + `sentence-transformers`.

## Commands

Run everything through Poetry so the project virtualenv is used.

| Task | Command |
|------|---------|
| Install deps | `poetry install` |
| Lint | `poetry run ruff check .` |
| Format | `poetry run ruff format .` |
| Type-check (strict) | `poetry run mypy churnbench` |
| Test | `poetry run pytest` |
| Run the CLI | `poetry run churnbench --help` |
| Bring up the fabric | `cd infra && docker compose up` |
| Tear down the fabric | `cd infra && docker compose down -v` |

**Before opening a PR, all four must pass:** `ruff check`, `ruff format --check`,
`mypy churnbench` (strict), and `pytest` (316 tests).

## Layout

```
churnbench/
  cli.py              # Typer CLI entry point
  fabric/             # Clients for Postgres / Mongo / SaaS API / docs + projector
  generator/          # timeline.py — synthetic drift timeline generation
  ledger/             # ledger.py — ground-truth mutation ledger
  tasks/              # Task schema, generator, ledger resolver (gold answers)
  arms/               # Agent "arms" under evaluation + grounding internals
  eval/               # Harness, scoring, parsing
infra/
  docker-compose.yml  # postgres + mongodb + saas_api
  api/                # FastAPI mock SaaS (intentionally slow + rate-limited)
  postgres/init.sql   # Warehouse schema seed
documentation/        # Engineering docs — read before changing the harness
assets/               # Committed figures
tests/                # pytest suite
```

## Read this before changing the harness or interpreting results

[`documentation/`](documentation/) covers the parts of this system that are
non-obvious and have caused real errors:

- [HARNESS.md](documentation/HARNESS.md) — lifecycle, config hashing,
  checkpointing, retry, guards
- [SCORING.md](documentation/SCORING.md) — verdicts, `t_eff`, tolerances
- [DATA_INTEGRITY.md](documentation/DATA_INTEGRITY.md) — auditing runs, the
  targeted-retry protocol, secret scanning
- [OPERATIONS.md](documentation/OPERATIONS.md) — running experiments, API
  instability, rerun checklist

Three traps worth knowing up front:

1. **The refresh scheduler bounds cache age by TTL, not window length.** An entity
   with `ttl_days=1` is 0–2 days old at `T` whether the window is 1 day or 29.
2. **A failed API call becomes a scored answer, not a crash.** The
   retry-exhaustion stub (`answer_raw="reasoning_error"`) is invisible in the
   summary. Audit before trusting a run.
3. **`config_hash` excludes the git SHA and model client config.** Re-running the
   same window under different code overwrites the previous result. Back up first.

## Conventions

- **Ruff** line length 100, target `py311`. **Mypy** is `strict = true` — new code
  must be fully typed; do not add `# type: ignore` without a reason comment.
- Keep secrets out of git. Real credentials belong in `.env` (git-ignored); only
  local-dev defaults (e.g. the Postgres `churn` password) live in `docker-compose.yml`.
- Match surrounding style; prefer `pydantic` models for structured data.

### What is and is not committed

`data/` and `docs/` are generated output and gitignored — `docs/` is the contract
corpus that `Projector.project_docs()` wipes and rebuilds on every run, **not** a
documentation directory. Engineering docs live in `documentation/`.

Deliberate exceptions use explicit gitignore negation rather than `git add -f`:

| path | why |
|---|---|
| `data/full/ledger.jsonl` | ground truth every gold answer derives from |
| `results/*_full.json` | complete per-task result sets |
| `results/summary_*.json` | lightweight aggregate summaries |
| `results/supplementary/*.csv` | freshness-error tables |
| `paper/churnbench-aixse.pdf` | current submission draft |

Still gitignored: `*.traces.jsonl`, `*.ckpt.json`, `.env`, LaTeX build artifacts,
`paper/ieeeaccess.cls` (not freely redistributable).

Follow the negation pattern for anything new — it keeps the policy readable in one
place.

## Git & PRs

- Repo: `vsingh45/churnbench` (**public**). Default branch: `main`.
- Branch and open a PR for code changes: `feat/…`, `fix/…`, `docs/…`, `chore/…`.
  Result data and documentation have been committed directly to `main` during
  active experiment runs, at the maintainer's direction — ask before assuming
  either workflow.
- Use the `/open-pull-request` skill to open a PR and `/review-pull-request` to
  review one. Both live in `.claude/skills/`.
- End commit messages with the required trailer, e.g.:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## Working with results

Result files are named `<arm>__<config_hash>.json`. Before citing any number from
one, confirm it is uncontaminated:

```bash
poetry run python3 -c "
import json; from pathlib import Path
for p in sorted(Path('results').glob('*_full.json')):
    d = json.loads(p.read_text())
    stubs = [r['task_id'] for r in d['results'] if r.get('answer_raw') == 'reasoning_error']
    print(f\"{p.name}: fe={d['summary']['n_freshness_error']} stubs={len(stubs)}\")
"
```

Active `*_full.json` files should report zero stubs. `*_BACKUP*` files are the
audit trail and are *expected* to show their original contamination — do not
delete them.
