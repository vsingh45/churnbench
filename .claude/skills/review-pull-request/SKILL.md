---
name: review-pull-request
description: >-
  Review an open GitHub pull request in the ChurnBench repo. Use when the user says
  "review PR #12", "look over this pull request", "is this PR safe to merge", or pastes
  a churnbench PR URL. Fetches the diff and CI checks via gh, reviews for correctness,
  typing/strictness, freshness-ledger integrity, secrets, and test coverage, then leaves
  a structured review (approve or request changes). Do NOT use to open a new PR (use
  open-pull-request) or to merge without the user's explicit go-ahead.
metadata:
  author: Vivek Kumar Singh
  version: 1.0.0
  mcp-server: none
license: MIT
compatibility: Requires git and the GitHub CLI (gh), authenticated for vsingh45/churnbench.
---

# Review Pull Request

Review an open ChurnBench PR and leave a structured, actionable review. Drives `gh`.

## Instructions

### Phase 1 — Locate the PR
- From a number: `gh pr view <n> --json number,title,headRefName,baseRefName,state,url,body`.
- From a URL: pass it to `gh pr view <url> --json ...`.
- **Stop** if the PR is already merged or closed — report state and ask whether to continue.

### Phase 2 — Gather context
```
gh pr diff <n>                 # the change
gh pr checks <n>               # CI status
gh pr view <n> --json files    # touched files
```
Read the diff in full. For non-trivial logic, open the surrounding files for context.

### Phase 3 — Review against ChurnBench concerns
Check, in roughly this priority:
1. **Correctness** — does the change do what the PR claims? Edge cases, off-by-one,
   async misuse (`pytest-asyncio` / `httpx`), incorrect Postgres/Mongo queries.
2. **Ledger & freshness integrity** — any mutation to the fabric must be recorded in the
   ground-truth ledger; scoring must compare against world state at time `T`, not "now".
   Flag changes that could desync the ledger from the fabric.
3. **Typing** — project is `mypy --strict`; new code must be fully typed, no bare
   `# type: ignore`.
4. **Secrets** — no real credentials, API keys, or tokens in the diff; only local-dev
   defaults belong in `docker-compose.yml`. No generated `data/` committed.
5. **Tests** — new behavior needs coverage under `tests/`. Note missing tests.
6. **Style** — ruff line length 100; consistent with surrounding code.

### Phase 4 — Leave the review
Decide: **approve** if clean, **request changes** if any blocking issue, **comment** if only
non-blocking nits. Post it:
```
gh pr review <n> --approve --body "<summary>"
# or
gh pr review <n> --request-changes --body "<summary + itemized blocking issues>"
# or
gh pr review <n> --comment --body "<non-blocking observations>"
```
Confirm the destructive/outward step with the user before posting if unsure — a review is a
public action on the PR. **Never merge** as part of this skill; if asked, confirm explicitly
first, then `gh pr merge`.

## Output Contract

The review body must contain:
- **Verdict** — approve / request changes / comment, one line.
- **Blocking** — numbered list of must-fix issues with `file:line` anchors (empty if none).
- **Non-blocking** — optional nits and suggestions.
- **CI** — summary of `gh pr checks` (passing/failing).

## Examples

- "Review PR #7." → fetch diff + checks, review, `gh pr review 7 --approve` or
  `--request-changes` with itemized issues.
- "Is this PR safe to merge? <url>" → full review; verdict + CI summary; do not merge.
- "Just leave comments, don't block." → use `gh pr review --comment`.

## Performance Notes

- `gh pr diff` is the primary signal; only open full files when the diff lacks context.
- Run `gh pr checks` early — if CI is red, that often frames the review (don't approve over
  failing required checks).
- Anchor every blocking issue to `file:line` so the author can jump straight to it.

## Troubleshooting

- **`gh: not authenticated`** → `gh auth status`; private repo needs a `repo`-scoped token.
- **`no pull requests found`** → confirm the number/URL and that it targets `vsingh45/churnbench`.
- **Diff too large to reason about** → review file-group by file-group; summarize per area.
- **Checks pending** → note "CI still running" in the review rather than assuming pass.
- **Asked to merge** → confirm explicitly, verify checks are green and the PR is approved,
  then `gh pr merge <n> --squash` (default to squash unless told otherwise).
