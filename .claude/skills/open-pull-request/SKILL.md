---
name: open-pull-request
description: >-
  Open a well-formed GitHub pull request for the ChurnBench repo. Use when the user
  says "open a PR", "raise a pull request", "push this for review", or after finishing
  a change on a feature branch. Runs the local quality gate (ruff, mypy, pytest) first,
  creates a branch off main if still on main, commits with the required trailer, pushes,
  and opens the PR via gh with a filled-out description. Do NOT use for reviewing an
  existing PR (use review-pull-request) or for merging.
metadata:
  author: Vivek Kumar Singh
  version: 1.0.0
  mcp-server: none
license: MIT
compatibility: Requires git and the GitHub CLI (gh), authenticated for vsingh45/churnbench.
---

# Open Pull Request

Open a reviewable GitHub PR for ChurnBench. This skill drives `git` + `gh` (the repo
lives on GitHub, not Harness Code), and enforces the project quality gate before pushing.

## Instructions

Follow these phases in order. Stop and ask the user if a stop condition is hit.

### Phase 1 — Scope check
1. Confirm the working directory is the churnbench repo (`git rev-parse --show-toplevel`).
2. `git status --short` to see what will be committed. If there are unrelated changes,
   ask the user which to include. **Stop** if the tree is clean and nothing is staged —
   there is nothing to open a PR for.

### Phase 2 — Branch
1. `git branch --show-current`. If it is `main`, create a branch — never commit to `main`.
   Derive the name from the change: `feat/…`, `fix/…`, `docs/…`, or `chore/…`.
   `git switch -c <branch>`.
2. If already on a feature branch, keep it.

### Phase 3 — Quality gate (all must pass)
Run and show output. **Stop and report** on any failure — do not push broken code.
```
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy churnbench
poetry run pytest
```
If the user explicitly says to skip a slow step (e.g. pytest), note it in the PR body.

### Phase 4 — Commit
1. `git add -A` (or only the agreed files).
2. Commit with a concise subject + body and the required trailer:
   ```
   <type>: <imperative summary>

   <what changed and why>

   Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
   ```

### Phase 5 — Push & open PR
1. `git push -u origin <branch>`.
2. Open the PR against `main` with a filled description:
   ```
   gh pr create --base main --head <branch> \
     --title "<type>: <summary>" \
     --body "<see contract below>"
   ```
3. Return the PR URL to the user. **Do not merge** — opening a PR is where this skill stops.

## Output Contract

The PR body must contain:
- **Summary** — 1–3 sentences on what changed and why.
- **Changes** — bullet list of the notable edits.
- **Testing** — which gate commands were run and their result (pass/skip).
- **Notes** — follow-ups, risks, or skipped checks, if any.

Final chat response must include the PR URL and a one-line summary of the gate results.

## Examples

- "Open a PR for the freshness-error scorer I just wrote." → branch `feat/freshness-scorer`,
  run gate, commit, push, `gh pr create`, return URL.
- "Raise a pull request for these docs fixes." → branch `docs/readme-fixes`, gate
  (ruff/mypy/pytest still run; trivially pass for docs), open PR.
- "Push this for review but skip the slow tests." → run ruff + mypy, note the skipped
  pytest in the PR body's Testing section, open PR.

## Performance Notes

- The quality gate is the slow part; `pytest` dominates. Run `ruff` and `mypy` first so
  fast failures surface before the test suite.
- `gh pr create` fails if the branch has no commits ahead of `main` — Phase 4 guarantees
  at least one commit, so never call it on an unpushed/empty branch.
- Prefer `git switch -c` over `git checkout -b` for clarity; both work.

## Troubleshooting

- **`gh: not authenticated`** → run `gh auth status`; the repo is private, so a token
  with `repo` scope for `vsingh45` is required. Ask the user to `gh auth login`.
- **`ruff format --check` fails** → run `poetry run ruff format .`, re-stage, re-commit.
- **`mypy` errors on new code** → the project is `strict`; add proper types rather than
  `# type: ignore`. Only ignore with an inline reason if a dependency lacks stubs.
- **Push rejected (branch protection on `main`)** → confirm you are on a feature branch,
  not `main`; the PR is the only path to `main`.
- **PR already exists for this branch** → `gh pr view` to fetch and return the existing URL
  instead of erroring.
