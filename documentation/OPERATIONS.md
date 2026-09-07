# Operations: Running Experiments

Practical guidance for running ChurnBench experiments, including failure modes
that cost real hours during this project.

---

## CLI surface

| command | purpose |
|---|---|
| `generate` | Run the timeline simulator, persist the ground-truth ledger |
| `snapshot` | Project the ledger onto Postgres, Mongo and docs at timestamp `T` |
| `tasks` | Generate evaluation tasks from the ledger frozen at `T` |
| `run` | Run one arm against the snapshot and score results |
| `score` | Load all results from a run directory, generate paper tables |
| `smoke` | 5-task end-to-end check with real API calls |

```bash
poetry run churnbench run <arm> <run_dir> --t-prime <ISO> --t <ISO> [--seed 42]
```

Arms: `naive`, `classic_rag`, `hierarchical`, `grounding`, and the ablations
`grounding_no_freshness_tiers`, `grounding_no_semantic_model`,
`grounding_no_source_routing`.

Before opening a PR: `ruff check`, `ruff format --check`, `mypy churnbench`
(strict), `pytest` — all four must pass.

---

## Experiment design: hold `T` fixed, vary `T′`

Every task in `data/full/tasks.jsonl` carries `T = 2024-04-27`, and the harness
asserts `task.T == harness T` ([HARNESS.md](HARNESS.md#guard-taskt-must-equal-the-harness-t)).
So the cache-age axis is swept by moving `T′`:

| window | `--t-prime` | `--t` | cache age |
|---|---|---|---|
| D+1 | `2024-04-26` | `2024-04-27` | 1 day |
| D+14 | `2024-04-13` | `2024-04-27` | 14 days |
| D+28 | `2024-03-29` | `2024-04-27` | 29 days |

The earlier design — fixing `T′` and moving `T` — routed every hot/warm entity to
a live origin at short windows and made freshness errors structurally impossible.
The guard now rejects that configuration outright.

---

## Provider and model configuration

Selected by environment variable, read in `churnbench/arms/base.py`:

```bash
CHURNBENCH_PROVIDER=nvidia_nim   # default
CHURNBENCH_PROVIDER=anthropic
CHURNBENCH_MODEL=<override>      # optional, either provider
```

| provider | default model | endpoint |
|---|---|---|
| `nvidia_nim` | `nvidia/nemotron-3-ultra-550b-a55b` | `https://integrate.api.nvidia.com/v1` |
| `anthropic` | `claude-sonnet-4-6` | Anthropic API |

Credentials live in `.env` (gitignored): `NVIDIA_API_KEY`, `ANTHROPIC_API_KEY`.
All arms call the same `llm()` helper, so provider and sampling are never a
confound across arms.

### Reasoning mode is off, deliberately

Nemotron-3 models default to **reasoning ON**, which wraps every completion in
chain-of-thought. For this benchmark's three call types — structured JSON
need-resolution, occasional SQL generation, short answer synthesis — that is pure
overhead. Observed with it on: SQL-generation calls returning 1,000–1,500 output
tokens at 8–13 tok/s, i.e. 110–121s per call, for what should be a ~30-token
SQL string.

`llm()` disables it:

```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

Verified: the same prompt returns 33 output tokens in ~25s instead of 1,000+ in
110s+. This is NVIDIA's documented toggle and applies across the Nemotron-3
family.

**This changes results, not just speed.** Runs made before this fix are not
config-matched to runs made after it. Compare only within one setting, and use a
pinned worktree if you must retry tasks in an older run
([DATA_INTEGRITY.md §3](DATA_INTEGRITY.md#step-3--verify-config-identity)).

---

## Runtime expectations

A 180-task run against NIM takes roughly **2–3.5 hours** when the API is healthy.
Most tasks resolve in milliseconds — 94% of retrievals hit a template and never
call the model for SQL. Wall-clock is dominated by the two per-task LLM calls plus
the minority of tasks that fall through to LLM-generated SQL.

The `_schedule_refreshes` walk adds only ~2 minutes even across a 29-day window,
despite re-projecting the fabric on each due day. It is not the bottleneck.

Do not run two experiments concurrently. The Postgres/Mongo fabric is shared and
re-projected in place; concurrent runs corrupt each other's view of the data.
Queue them.

---

## API instability

NIM was unreliable for extended stretches during this project. Symptoms, in
rough order of frequency:

| error | meaning |
|---|---|
| `429 RateLimitError` | account/key quota exhausted — the dominant failure |
| `503 Service temporarily overloaded` | transient backend load |
| `404 NotFoundError` | model function-id lookup failed; intermittent, not fatal |
| `APIConnectionError` | network |

At worst, 102 of 180 tasks hit at least one error and 24 exhausted all retries.
Practical guidance:

- **Check the checkpoint, not the log.** Progress logs are buffered through
  `grep | tee` and lag badly; `PYTHONUNBUFFERED=1` helps. The `.ckpt.json` is
  written directly and is always current.
- **Watch for permanent failures during the run**, not after. Every
  `attempt 3/3` line is a task that will land as a stub.
- **A fresh key resolves quota exhaustion immediately** — and the old key's
  throttling also manifests as *slowness*, not only errors. Per-call latency for
  identical prompts dropped from 110–121s to 15–18s on a fresh key.
- **Rotate any key that has appeared in a terminal, log, or transcript.**

If a run completes with stub failures, repair it with the targeted-retry protocol
rather than rerunning the whole experiment
([DATA_INTEGRITY.md §3](DATA_INTEGRITY.md#3--the-targeted-retry-protocol)).

---

## Before a rerun that reuses a config hash

1. Compute the hash and check whether that file already exists
   ([DATA_INTEGRITY.md §6](DATA_INTEGRITY.md#6--config-hash-collisions)).
2. Back it up if it is still needed — the rerun overwrites it.
3. **Delete the existing `.traces.jsonl`.** It is opened in append mode; a rerun
   otherwise produces a file with duplicate `task_id` entries from two runs.
4. **Delete any stale `.ckpt.json`.** Resuming onto a checkpoint written by
   different code produces a result mixing both behaviours.

---

## Repository conventions

Committed result data, so reviewers can verify ground truth and not just
arithmetic:

| path | contents |
|---|---|
| `data/full/ledger.jsonl` | ground-truth ledger, every gold answer derives from it |
| `results/*_full.json` | complete per-task result sets, incl. `retry_provenance` |
| `results/summary_*.json` | lightweight aggregate summaries |
| `results/supplementary/*.csv` | complete freshness-error tables, per run and combined |

Gitignored: `data/` (except the ledger and its README), `docs/` (generated
contract corpus), `*.traces.jsonl`, `*.ckpt.json`, `.env`.

The `.gitignore` uses explicit negation for deliberate exceptions
(`!data/full/ledger.jsonl`, `!results/*_full.json`, `!paper/churnbench-aixse.pdf`)
rather than force-adding. Follow that pattern — it keeps the policy readable.
