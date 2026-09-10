"""Extract observed per-entity cache ages (Table II) from run traces.

Source traces are gitignored (results/**/*.traces.jsonl) -- this script's
OUTPUT csv is the committed, traceable artifact. Re-running the D+28 matched
pair (see documentation/OPERATIONS.md) regenerates the source traces this
script reads, so the number remains independently reproducible even though
the raw traces themselves are not checked in.

Run: poetry run python3 scripts/export_cache_age_table.py
"""
import csv
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from churnbench.arms.grounding.semantic_model import ENTITY_REGISTRY  # noqa: E402

RESULTS_DIR = PROJ / "data/full/results"
OUT = PROJ / "results/supplementary/table2_cache_age.csv"

# (run_id, config label, trace filename, T)
RUNS = [
    ("D28_tiered", "grounding__6326534d90587f84", "grounding__6326534d90587f84.traces.jsonl", date(2024, 4, 27)),
    ("D28_untiered", "grounding_no_freshness_tiers__1d64ea6fe73b4de3",
     "grounding_no_freshness_tiers__1d64ea6fe73b4de3.traces.jsonl", date(2024, 4, 27)),
]

COLUMNS = ["run_id", "config", "entity_class", "tier", "ttl_days", "last_refresh", "T", "age_days"]


def extract_observed_ages(trace_path: Path, T: date) -> dict[str, set[str]]:
    """Return {entity_class: {last_refresh values observed in staged retrievals}}."""
    observed: dict[str, set[str]] = defaultdict(set)
    for line in trace_path.read_text().splitlines():
        d = json.loads(line)
        for e in d["trace"]:
            if e.get("role") == "retrieval" and e.get("staged_vs_live") == "staged" and e.get("last_refresh"):
                observed[e["entity_class"]].add(e["last_refresh"])
    return observed


def main() -> None:
    rows = []
    missing_traces = []

    for run_id, config, trace_fname, T in RUNS:
        trace_path = RESULTS_DIR / trace_fname
        if not trace_path.exists():
            missing_traces.append(str(trace_path))
            continue
        observed = extract_observed_ages(trace_path, T)
        for ec_name, ec in ENTITY_REGISTRY.items():
            if ec_name not in observed:
                continue  # not observed in a staged retrieval this run
            for lr_str in sorted(observed[ec_name]):
                lr = date.fromisoformat(lr_str)
                age_days = (T - lr).days
                rows.append({
                    "run_id": run_id,
                    "config": config,
                    "entity_class": ec_name,
                    "tier": ec.tier,
                    "ttl_days": ec.ttl_days if ec.ttl_days is not None else "",
                    "last_refresh": lr_str,
                    "T": T.isoformat(),
                    "age_days": age_days,
                })

    if missing_traces:
        print("WARNING: could not find these trace files (run the corresponding "
              "experiment to regenerate them -- see documentation/OPERATIONS.md):")
        for m in missing_traces:
            print(f"  {m}")
        if not rows:
            print("No rows written -- no source traces available.")
            return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} rows written to {OUT.relative_to(PROJ)}")
    for r in rows:
        print(f"  {r['run_id']:<14} {r['entity_class']:<24} tier={r['tier']:<5} "
              f"ttl={r['ttl_days']!s:<3} last_refresh={r['last_refresh']} age_days={r['age_days']}")


if __name__ == "__main__":
    main()
