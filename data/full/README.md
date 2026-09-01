# data/full — ground-truth ledger

`ledger.jsonl` is the ground-truth mutation ledger every gold answer in the
paper's Design-B sweep derives from (via `LedgerResolver`, at whatever
timestamp a task's `gold(resolver, at=T)` call asks for). It's committed here
so reviewers can audit the paper's *data*, not just the arithmetic recorded
in `results/*_full.json` and `results/supplementary/*.csv`.

## Generation

```
poetry run churnbench generate --days 180 --seed 42 --out data/full --start 2024-01-01
```

This runs `TimelineSimulator` (`churnbench/generator/timeline.py`) with a
seeded RNG and writes one JSON object per event, in emission order, to
`ledger.jsonl`. The command above uses every CLI default except `--out`, so
it's equivalent to `poetry run churnbench generate --out data/full`.

- **Seed**: 42
- **Start date**: 2024-01-01
- **Duration**: 180 days (last event date: 2024-06-28)
- **Event count**: 56,370

## Determinism

Regenerating from the command above reproduces `ledger.jsonl` **byte-for-byte**
(verified via `diff` and matching MD5 checksums before this file was
committed). The simulator has no non-seeded randomness or wall-clock
dependence — same seed, same output, always.

## Not committed here

`tasks.jsonl` (the 180-task benchmark set derived from this ledger, seed 7),
`docs/` (the generated contract corpus), and `results/` (raw per-run harness
output) all remain gitignored under `data/full/` — they're regenerable from
this ledger plus the harness/generator code and aren't needed to audit the
gold-answer ground truth itself.
