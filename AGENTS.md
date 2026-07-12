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
- Agentic layer: **LangGraph / LangChain** (+ `langchain-anthropic`).
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
`mypy churnbench` (strict), and `pytest`.

## Layout

```
churnbench/
  cli.py              # Typer CLI entry point
  fabric/             # Clients for Postgres / Mongo / SaaS API / docs
  generator/          # timeline.py — synthetic drift timeline generation
  ledger/             # ledger.py — ground-truth mutation ledger
  tasks/              # Benchmark task definitions
  arms/               # Agent "arms" under evaluation
  eval/               # Scoring: freshness error, correctness
infra/
  docker-compose.yml  # postgres + mongodb + saas_api
  api/                # FastAPI mock SaaS (intentionally slow + rate-limited)
  postgres/init.sql   # Warehouse schema seed
tests/                # pytest suite
```

## Conventions

- **Ruff** line length 100, target `py311`. **Mypy** is `strict = true` — new code
  must be fully typed; do not add `# type: ignore` without a reason comment.
- Keep secrets out of git. Real credentials belong in `.env` (git-ignored); only
  local-dev defaults (e.g. the Postgres `churn` password) live in `docker-compose.yml`.
- `data/` is git-ignored except `data/.gitkeep`. Do not commit generated fabric data.
- Match surrounding style; prefer `pydantic` models for structured data.

## Git & PRs

- Repo: `vsingh45/churnbench` (**private**). Default branch: `main`.
- Never commit directly to `main`. Branch, commit, open a PR.
- Branch names: `feat/…`, `fix/…`, `docs/…`, `chore/…`.
- Use the `/open-pull-request` skill to open a PR and `/review-pull-request` to
  review one. Both live in `.claude/skills/`.
- End commit messages with the required trailer:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
