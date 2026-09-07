# The Evaluation Harness

`churnbench/eval/harness.py` — `RunHarness` orchestrates one arm's complete
lifecycle for one experiment. This document covers what it does, the invariants
it maintains, and the hazards that have actually bitten during real runs.

---

## The six-step lifecycle

`RunHarness.run()` executes these in order. Every step matters to the result;
skipping or reordering them silently changes what the experiment measures.

| # | Step | What it does |
|---|---|---|
| 1 | `_project_at(ledger, T_prime)` | Wipe-and-rebuild Postgres, Mongo and docs to the world state at `T′` |
| 2 | `arm.setup(config, T_prime)` | Arm builds its indexes and staged caches. Staleness is *born* here |
| 3 | `_schedule_refreshes(arm, ledger, T_prime, T)` | Walk `T′+1 … T` day by day, re-running ETL for entities whose TTL lapsed |
| 4 | `_project_at(ledger, T)` | Bring the fabric to the evaluation timestamp |
| 5 | `arm.answer(task)` per task | Checkpoint every 10 tasks |
| 6 | Score + write | Per-task verdict, config-hash-named JSON output |

Step 3 is the only place the tiered-staleness lifecycle is exercised end to end
against real projected data. The paper's central claim depends on it being
correct.

### Why step 3 has the property it does

The walk is **day by day**, not a single catch-up refresh at the end:

```python
day = T_prime + timedelta(days=1)
while day <= T:
    due = refresh_due(arm._registry, day)
    if due:
        self._project_at(ledger, day)     # re-project so ETL reads that day's data
        for ec in due:
            refresh_entity(...)
            ec.last_refresh = day
    day += timedelta(days=1)
```

Because it checks *every* day, any entity whose `ttl_days` is shorter than the
window gets refreshed repeatedly and ends up bounded under its TTL at evaluation
time — **regardless of how far back `T′` is**. An entity with `ttl_days=1`
refreshes every 2 days and is 0–2 days old at `T`, whether the window is 1 day or
29 days.

This is not a bug; it is the mechanism the benchmark measures. But it means
**cache age at `T` is not a function of `T − T′`** for any entity whose TTL fits
inside the window. See [ARCHITECTURE.md §3](../ARCHITECTURE.md).

### `no_freshness_tiers` skips the walk entirely

```python
if not isinstance(arm, GroundingArm):
    return
if arm.no_freshness_tiers:
    return
```

The second guard exists because `router.decide()` already forces
`is_stale = False` for this ablation. Without the guard, the scheduler would keep
auto-healing `last_refresh` in the background, and the ablation would be a silent
no-op — the cache would in fact stay current while the arm reported serving
unrefreshed `T′`-era data. Both halves have to agree: routing ignores staleness
**and** the cache genuinely never refreshes.

---

## Config hashing

```python
payload = {
    "arm": arm_name,
    "flags": arm_flags,
    "model": model,
    "T_prime": T_prime.isoformat(),
    "T": T.isoformat(),
    "seed": seed,
    "task_hash": sha256(sorted(task_ids))[:16],
}
config_hash = sha256(json.dumps(payload, sort_keys=True))[:16]
```

Results are written to `<arm>__<config_hash>.json`, which makes reruns of the
same configuration idempotent by construction.

### What is NOT in the hash

This has caused real data loss twice, so it is worth stating plainly. The hash
does **not** include:

- **the git SHA** — code changes do not produce a new hash
- **`enable_thinking`** and any other client-level model config
- **the API key or provider**

**Consequence:** re-running the same `(arm, T′, T, seed, tasks)` under *different
code* computes the **same hash** and **overwrites** the previous result file. If
the previous result is one you still need — a pre-fix baseline, a run at a
different reasoning mode — back it up before rerunning. Both the
`*_PREFIX_BASELINE_BACKUP.*` and `*_PRE_RETRY_BACKUP.*` files in this repo exist
because of exactly this.

The CLI also does not pass `arm_flags` through, so ablation arms are distinguished
by `arm_name` alone (`grounding` vs `grounding_no_freshness_tiers`), which is
sufficient — but means `flags` is always `{}` in practice.

---

## Checkpointing and resume

Checkpoints are written every `_CHECKPOINT_EVERY = 10` tasks to
`<arm>__<hash>.ckpt.json`, and deleted on successful completion.

On resume, completed task IDs are skipped — but **steps 1–4 run in full first**.
The fabric is re-projected, the arm is re-set-up, and the refresh walk re-executes
before any pending task is answered. This is correct (the arm needs its cache and
the fabric needs to be at `T`), but it means resuming is not cheap, and it means a
checkpoint from a run under *different code* will produce a result file mixing
old and new behaviour.

**Delete stale checkpoints before a clean rerun.** A contaminated D+14 result
early in this project came from resuming onto a pre-fix checkpoint.

---

## Retry policy

The built-in policy is deliberately modest:

```python
_ANSWER_MAX_RETRIES = 2      # 3 attempts total
_ANSWER_RETRY_BASE_S = 15    # waits 15s, then 30s
```

On permanent failure, `_answer_with_retry()` returns a **stub** `ArmResult`:

```python
ArmResult(
    answer_raw="reasoning_error",
    answer_parsed=None,
    ...,
    trace=[{"role": "error", "message": err_msg}],
)
```

This stub is the single most important thing to know about when auditing results.
It is scored like any other answer, so it lands in `reasoning_error` or
`parse_failure` depending on the task's `answer_type` — it does **not** appear as
a distinct failure category in the summary. See
[DATA_INTEGRITY.md](DATA_INTEGRITY.md) for how to detect it.

Against an unstable API this policy is not enough. Targeted-retry passes in this
project used 7 attempts with exponential backoff (20s → 180s, capped) and a 180s
per-request timeout, which drove permanent failures to zero in every case.

---

## Guard: `task.T` must equal the harness `T`

```python
mismatched = [t.task_id for t in tasks if t.T != T]
if mismatched:
    raise AssertionError(...)
```

The grounding arm routes using `task.T` (via `_execute_retrieval(need, task.T, ...)`),
while gold answers are resolved at the harness `T`. If these diverge, staleness
routing and gold resolution use different timestamps and **freshness errors are
mis-classified**.

This guard exists because the original sweep design varied the harness `T` while
every task carried a fixed `task.T = 2024-04-27`. At D+1 that made every hot/warm
entity appear 29 days stale to the router, routing everything to live origins and
making freshness errors structurally impossible. The experiment was measuring
nothing.

The fix was to hold `T` fixed at the tasks' own `T` and vary `T′` instead
(Design B). The guard makes the broken configuration fail loudly instead of
silently producing publishable-looking numbers.

---

## Trace files are opened in append mode

```python
with trace_path.open("a") as tf:
    tf.write(json.dumps({"task_id": ..., "trace": ...}) + "\n")
```

A rerun that hits an existing `*.traces.jsonl` **appends to it** rather than
truncating, producing a file with duplicate `task_id` entries from two different
runs. Delete or move the trace file before any rerun that reuses a config hash.

---

## Output layout

```
results/<run>/<arm>__<cfg_hash>.json          summary + per-task (committed as *_full.json)
results/<run>/<arm>__<cfg_hash>.traces.jsonl  full traces (gitignored)
results/<run>/<arm>__<cfg_hash>.ckpt.json     checkpoint (deleted on success)
```

Result JSON top level:

| field | notes |
|---|---|
| `arm`, `config_hash`, `git_sha` | `git_sha` added so results are traceable to code |
| `t_prime`, `t` | ISO dates |
| `n_tasks`, `summary` | `SummaryMetrics` as a dict |
| `results[]` | one `TaskResult` per task |
| `retry_provenance` | present only on merged results — see [DATA_INTEGRITY.md](DATA_INTEGRITY.md) |

Note that `TaskResult` does **not** carry the multi-step trace; that lives only in
the sibling `.traces.jsonl`, which is gitignored. Anything a reviewer must be able
to verify has to be in the result JSON itself.

---

## Module sizes

| module | LOC | role |
|---|---|---|
| `arms/` | 3,739 | the four arms + grounding internals |
| `eval/` | 1,096 | harness, scoring, parsing |
| `tasks/` | 925 | generator, resolver, schema |
| `fabric/` | 810 | Postgres / Mongo / SaaS / docs clients, projector |
| `generator/` | 325 | timeline simulator |
| `ledger/` | 287 | ground-truth mutation log |
| `cli.py` | 335 | Typer entry point |

316 tests.
