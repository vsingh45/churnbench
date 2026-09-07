# Scoring and the Verdict Taxonomy

`churnbench/eval/scoring.py` — how an answer becomes a verdict, and why the
benchmark needs two reference answers instead of one.

---

## The four verdicts

```
correct          answer matches ground truth at T
freshness_error  wrong at T, but correct at t_eff — a stale cache served old truth
reasoning_error  wrong at both T and t_eff — model failure, not a staleness artifact
parse_failure    output could not be coerced to the task's answer_type
```

`classify_verdict()` evaluates them in that order:

```python
pr = parse(answer_raw, answer_type)
if pr.parse_failure:                       return "parse_failure"
if is_correct(pr.value, gold_at_T):        return "correct"
if t_eff < T and is_correct(pr.value, gold_at_t_eff):
                                           return "freshness_error"
return "reasoning_error"
```

Two properties follow from the ordering that matter when reading results:

- **`freshness_error` requires `t_eff < T`.** An arm that always reads live data
  has `t_eff == T` and can never produce one. For the `naive` arm this is a
  structural zero, not a measurement.
- **A stub failure can never be a freshness error.** The retry-exhaustion stub
  (`answer_raw="reasoning_error"`) fails to match either reference, so it always
  lands in `reasoning_error` or `parse_failure`. This is why API instability
  *deflates* accuracy without inflating freshness errors — see
  [DATA_INTEGRITY.md](DATA_INTEGRITY.md).

---

## Why two references

A single-reference benchmark sees a wrong answer and stops there. It cannot
distinguish:

- the model reasoned badly, from
- the model reasoned correctly over data that was true last week

Both are wrong at `T`. Only the second is a *grounding* failure, and only the
second is fixed by better cache scheduling rather than a better model. Scoring
against `gold@t_eff` as well as `gold@T` is what makes that distinction
mechanical rather than a judgement call.

`Task.gold(resolver, at=...)` computes ground truth at an arbitrary timestamp
directly from the ledger, so the second reference costs nothing but a resolver
call — no grader model, no contamination.

---

## `t_eff` — the effective retrieval time

One rule per arm:

| arm | `t_eff` | rationale |
|---|---|---|
| `naive` | `T` | always live; freshness errors structurally impossible |
| `classic_rag` | `T_prime` | the entire index is frozen at build time |
| `hierarchical` | `min(source_ts)` across workers; `"live"` maps to `T` | as stale as its stalest worker |
| `grounding` | `min(last_refresh)` across staged retrievals; live routes contribute `T` | as stale as its stalest cache hit |

For the grounding arm this reads the trace:

```python
for entry in trace:
    if entry.get("role") != "retrieval": continue
    if entry.get("staged_vs_live") == "staged":
        t_eff_dates.append(date.fromisoformat(entry["last_refresh"]))
    else:
        t_eff_dates.append(T)          # live routes are current by definition
return min(t_eff_dates) if t_eff_dates else T_prime
```

`staged_vs_live` is `"staged"` for `staged_sql` and `docs_index`, `"live"` for
everything else. A task that touches both a cache and a live source takes the
**minimum** — the answer is only as fresh as its stalest input.

A task with no retrieval entries at all falls back to `T_prime`, the conservative
choice.

---

## Correctness tolerances

`is_correct()` is type-aware. Exact equality is only used where it is meaningful:

| `answer_type` | rule |
|---|---|
| `int` | exact equality after `int()` coercion |
| `float` | 1% relative tolerance; exact-zero handled separately (`abs(a) < 1e-6`) |
| `list[str]` | set F1 ≥ 0.99, case-insensitive, whitespace-stripped |
| `str` | case-insensitive, whitespace-stripped equality |

**Use `is_correct()` when auditing, not `==`.** A naive equality check on floats
produces false negatives from representation noise — `1064.1` vs
`1064.1000000000001` is a match under the tolerance rule and a mismatch under
`==`. Three "failures" in an early audit of this project were exactly that
artifact.

---

## Summary metrics

`compute_summary()` returns counts and rates:

```
n_tasks, n_correct, n_freshness_error, n_reasoning_error, n_parse_failure
accuracy, freshness_error_rate, reasoning_error_rate, parse_failure_rate
mean_cost_usd, total_cost_usd, mean_latency_s
mean_input_tokens, mean_output_tokens
```

Field names in the JSON are the `n_*` forms — `n_freshness_error`, not
`freshness_error`. Analysis scripts that guess the short name silently read
zero.

---

## Freshness attribution

When a verdict is `freshness_error`, `stale_artifact_attribution()` records which
staged retrievals contributed:

```python
{"stale_artifacts": [
    {"entity_class": ..., "route": ..., "last_refresh": ...,
     "cache_miss_reason": ..., "measure": ...},
]}
```

This lands in the result's `attribution` field and is what makes it possible to
say *which tier* produced an error — the basis for the finding that hot-tier
entities contribute 25 of 45 errors with scheduling off and 0 of 4 with it on.

---

## Cost accounting

`cost_usd()` is the single place cost arithmetic happens. Embedding cost is always
$0 (local `sentence-transformers`), but `embedding_tokens` is still tracked so the
cost-decomposition table can report token counts per component. Unknown models
fall back to the Nemotron rate — a conservative over-estimate, never a silent
zero.
