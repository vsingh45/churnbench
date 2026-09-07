# Data Integrity: Auditing, Scanning, and Repair

Procedures for verifying that a result file measures what it claims to measure.
Every procedure here exists because the corresponding failure actually occurred
during this project's runs.

---

## 1. The silent failure mode

When `_answer_with_retry()` exhausts its attempts against an unstable API, it
returns a stub rather than crashing the run:

```python
ArmResult(
    answer_raw="reasoning_error",
    answer_parsed=None,
    input_tokens=0, output_tokens=0, cost_usd=0.0, latency_s=0.0,
    trace=[{"role": "error", "message": err_msg}],
)
```

That stub is scored like a real answer. It lands in `reasoning_error` or
`parse_failure` depending on the task's `answer_type`, and **nothing in the
summary distinguishes it from a genuine model failure**. A run can report a
plausible accuracy while a seventh of its tasks never reached the model at all.

### Why it biases in a specific direction

A stub matches neither `gold@T` nor `gold@t_eff`, so it can never be classified as
a `freshness_error`. API instability therefore:

- **deflates** `n_correct` and accuracy
- **inflates** `n_reasoning_error` and `n_parse_failure`
- leaves `n_freshness_error` **unchanged**

This was confirmed empirically five separate times in this project: every
targeted retry that eliminated stub failures left the freshness-error count
exactly where it was (4→4, 4→4, 7→7, 45→45, 7→7) while accuracy rose. If the
headline metric is freshness error, stub contamination is a threat to the
*accuracy* figure and a threat to *credibility*, but not to the headline number
itself.

---

## 2. Detecting stub failures

### Rigorous: trace signature

The stub is the only thing that produces a trace consisting of exactly one entry
with `role == "error"`:

```python
for line in traces_path.read_text().splitlines():
    t = json.loads(line)
    if [e.get("role") for e in t["trace"]] == ["error"]:
        stub_ids.append(t["task_id"])
```

A real answer always has `need_resolution` / `retrieval` / `synthesis` entries.
Use this when the `.traces.jsonl` is available.

### Proxy: `answer_raw`

```python
stubs = [r["task_id"] for r in d["results"] if r.get("answer_raw") == "reasoning_error"]
```

Necessary for committed result JSONs, since traces are gitignored. In practice
this has matched the trace-signature result exactly — no genuine model output has
ever been the literal string `reasoning_error` — but it is a proxy, and the
distinction is worth stating when the number matters.

### Sweep everything, not just the file you suspect

The one contaminated file discovered late in this project — 4 stub failures in
the D+28 ablation, the paper's headline result — had already been committed and
pushed. It was found only by sweeping *every* result file rather than the one
under active work. Audit the whole directory:

```bash
poetry run python3 -c "
import json; from pathlib import Path
for p in sorted(Path('results').glob('*_full.json')):
    d = json.loads(p.read_text())
    stubs = [r['task_id'] for r in d['results'] if r.get('answer_raw') == 'reasoning_error']
    print(f\"{p.name}: fe={d['summary']['n_freshness_error']} stubs={len(stubs)} {stubs or ''}\")
"
```

Backup files are *expected* to show their original contamination — that is what
they are for. Only the active `*_full.json` files should read zero.

---

## 3. The targeted-retry protocol

Re-running a whole experiment to fix a handful of API failures discards good data
and costs hours. The protocol below repairs only the failed tasks while leaving
every successful one byte-identical.

**Never skip a step.** Each guards against something that has gone wrong.

### Step 1 — Back up, verify the backup

```bash
cp results/<run>.json results/<run>.PRE_RETRY_BACKUP.json
cp results/<run>.traces.jsonl results/<run>.PRE_RETRY_BACKUP.traces.jsonl
md5 results/<run>.json results/<run>.PRE_RETRY_BACKUP.json   # must match
```

Backups are the audit trail. Do not delete them; commit them.

### Step 2 — Identify the failed tasks and why

List the task IDs and their terminal error from the run log
(`grep "attempt 3/3"`). Confirm the count matches the stub count from §2.

### Step 3 — Verify config identity

The retry must run under the **same code** as the original, or the merged result
mixes two configurations. Compare the result's `git_sha` against current `HEAD`:

```bash
python3 -c "import json;print(json.load(open('results/<run>.json'))['git_sha'])"
git rev-parse HEAD
```

If they differ, check what actually changed (`git diff <old> HEAD -- churnbench/`).
A change that cannot affect this arm is fine; a change that can is not.

**When the code has moved, pin it with a worktree** rather than checking out the
old commit in place:

```bash
git worktree add /tmp/pinned <old-sha>
# retry script does: sys.path.insert(0, "/tmp/pinned")
git worktree remove /tmp/pinned
```

This runs genuine old-commit code with zero risk to the working tree. It was
required for the D+14 retry, whose original run predated the `enable_thinking`
fix; without it, 5 retried tasks would have run with reasoning off while the
other 175 ran with it on.

### Step 4 — Reproduce the cache state, then answer only the failed tasks

Replay steps 1–4 of the harness lifecycle (project@T′, `arm.setup()`,
`_schedule_refreshes()`, project@T) so the registry's `last_refresh` values and
the fabric match the original run. Then answer **only** the failed task IDs, with
a hardened retry policy:

```
max_retries        6      (7 attempts)
backoff            exponential, 20s base, 180s cap
request timeout    180s
```

Write new traces to a **scratch file**, never the production trace path — that
one is opened in append mode.

### Step 5 — Merge, and prove you did not disturb anything

Take the untouched entries *directly from the backup object* — do not re-score or
re-serialize them, which risks float-representation drift. Then verify:

```python
for tid in successful_ids:
    assert json.dumps(backup_by_id[tid], sort_keys=True) == \
           json.dumps(merged_by_id[tid], sort_keys=True)
assert len(merged_results) == 180
```

Both assertions must pass before the merged file is written.

### Step 6 — Record provenance

The merged result carries a `retry_provenance` block so the repair is visible in
the data, not just in a commit message:

```json
{
  "retried_task_ids": ["task_0024", "..."],
  "retry_timestamp": "2026-09-07T...",
  "retry_config": {
    "max_retries": 6,
    "backoff": "exponential base=20s cap=180s",
    "request_timeout_s": 180,
    "git_sha_at_retry": "..."
  },
  "remaining_permanent_failures": [],
  "config_match_note": "..."
}
```

### Step 7 — Re-score and report the delta

Report before/after for every verdict count, not just the headline. A retry that
changes the freshness-error count is telling you something and deserves scrutiny;
one that does not is the expected outcome.

---

## 4. Secret scanning

Before making a repository public, or any time credentials have been handled:

```bash
git log -p --all | grep -icE 'nvapi-|sk-ant-'          # 0 = clean
git log -p --all | grep -c '<specific key fragment>'    # also 0
```

Check the **pipeline's** exit semantics carefully: `git log … | grep … | head`
reports `head`'s status, not `grep`'s. Use `grep -c` and read the count.

Scan staged files before committing, too — result JSONs contain raw model I/O and
are large enough that nobody reviews them by eye:

```bash
grep -c "nvapi-\|sk-ant-" results/*_full.json
```

If anything matches, **stop** — do not publish, and rotate the key.

---

## 5. What the `ttl_expired` counter cannot tell you

`cache_miss_reason` reads `ttl_expired` when the router falls through to a live
origin. It is a natural place to look for staleness. It is also a trap: it reads
**0 in both arms** of the tiering ablation, for opposite reasons.

- **Tiering ON** — the scheduler refreshes entities before their TTL lapses, so
  the router never sees a stale one.
- **Tiering OFF** — `router.decide()` forces `is_stale = False` and never
  evaluates the check, so it never reports a miss while serving 29-day-old data.

A metric that reports identical values for opposite behaviours is not measuring
the thing. The honest instrument is the observed `last_refresh` age per entity,
read from the traces.

---

## 6. Config-hash collisions

`config_hash` excludes the git SHA and all client-level model config
([HARNESS.md](HARNESS.md#config-hashing)). Re-running the same
`(arm, T′, T, seed, tasks)` under different code **overwrites** the earlier
result file in place.

Before any rerun that might collide:

```bash
poetry run python3 -c "
from datetime import date; from pathlib import Path
from churnbench.eval.scoring import config_hash
from churnbench.tasks.schema import load_tasks
tasks = load_tasks(Path('data/full/tasks.jsonl'))
print(config_hash('grounding', {}, 'nvidia/nemotron-3-ultra-550b-a55b',
                  date(2024,4,26), date(2024,4,27), 42, tasks))
"
ls results/ data/full/results/ | grep <that-hash>
```

If it exists, back it up first. The `*_PREFIX_BASELINE_BACKUP.*` files in this
repo exist because a matched-config rerun would otherwise have destroyed the
superseded baseline it was meant to be compared against.
