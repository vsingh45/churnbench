# ChurnBench Technical Documentation

Engineering documentation for the benchmark harness, scoring pipeline, and the
data-integrity procedures used to produce the paper's results.

| document | covers |
|---|---|
| [HARNESS.md](HARNESS.md) | The six-step lifecycle, refresh scheduling, config hashing, checkpointing, retry policy, guards, and the hazards each one exists to prevent |
| [SCORING.md](SCORING.md) | Verdict taxonomy, `t_eff` derivation per arm, correctness tolerances, freshness attribution, why the benchmark needs two reference answers |
| [DATA_INTEGRITY.md](DATA_INTEGRITY.md) | Detecting stub failures, the targeted-retry protocol, secret scanning, config-hash collisions, and one metric that cannot be trusted |
| [OPERATIONS.md](OPERATIONS.md) | CLI surface, experiment design, provider config, runtime expectations, API instability, rerun checklist |
| [architecture.html](architecture.html) | Interactive figures — system schematic, entity registry, staleness chart, results. Open in a browser |

See also [`../ARCHITECTURE.md`](../ARCHITECTURE.md) for the system overview with
GitHub-rendered diagrams, and [`../AGENTS.md`](../AGENTS.md) for coding-agent
onboarding.

---

## Where to start

**Running an experiment** → [OPERATIONS.md](OPERATIONS.md)

**Reading a result file** → [SCORING.md](SCORING.md) for what the verdicts mean,
then [DATA_INTEGRITY.md §2](DATA_INTEGRITY.md#2--detecting-stub-failures) to
confirm the run is not contaminated before you trust the numbers.

**Modifying the harness** → [HARNESS.md](HARNESS.md). The lifecycle ordering,
the config-hash inputs, and the `task.T == harness T` guard are all load-bearing;
each has a documented failure it prevents.

**Reproducing the paper's central result** →
[ARCHITECTURE.md §3](../ARCHITECTURE.md) for the claim, then
[OPERATIONS.md](OPERATIONS.md#experiment-design-hold-t-fixed-vary-t) for the four
commands that produce the 2×2.

---

## The short version

Three things about this codebase are non-obvious and cause real errors:

1. **The refresh scheduler bounds cache age by TTL, not by window length.** An
   entity with a 1-day TTL is 0–2 days old at evaluation time whether the window
   is 1 day or 29. This is the mechanism the benchmark measures, and it means
   "older cache" is not a synonym for "staler data."

2. **A failed API call becomes a scored answer, not a crash.** The retry-exhaustion
   stub lands in `reasoning_error` or `parse_failure` and is invisible in the
   summary. Always audit for it before trusting a run.

3. **`config_hash` ignores the git SHA and all model client config.** Re-running
   the same window under different code silently overwrites the previous result
   file. Back up first.
