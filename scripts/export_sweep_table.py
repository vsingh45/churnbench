"""Extract the three-window sweep (Table III) from committed results and traces.

correct / reasoning_errors / parse_failures / freshness_errors / accuracy_pct /
n_tasks come from the committed results/*_full.json summaries. ttl_lapses --
the count of retrievals where the router reported cache_miss_reason ==
"ttl_expired" -- requires the per-task trace and is extracted the same way as
export_cache_age_table.py (see that script's docstring re: source traces being
gitignored while this script's output is the committed artifact).

Run: poetry run python3 scripts/export_sweep_table.py
"""
import csv
import json
from pathlib import Path

PROJ = Path(__file__).resolve().parent
PROJ = PROJ.parent
RESULTS_DIR = PROJ / "data/full/results"
OUT = PROJ / "results/supplementary/table3_sweep.csv"

# (window, result json path (committed), trace filename (local, may be absent))
RUNS = [
    ("D+1",  PROJ / "results/d1_baseline_full.json",  "grounding__06e9fbdfe5d0b7d4.traces.jsonl"),
    ("D+14", PROJ / "results/d14_baseline_full.json", "grounding__31e6a01df61bf4d5.traces.jsonl"),
    ("D+28", PROJ / "results/d28_matched_baseline_full.json", "grounding__6326534d90587f84.traces.jsonl"),
]

COLUMNS = ["window", "t_prime", "T", "freshness_errors", "accuracy_pct", "ttl_lapses",
           "correct", "reasoning_errors", "parse_failures", "n_tasks",
           "git_sha", "reasoning_mode_note"]


def count_ttl_lapses(trace_path: Path) -> int | None:
    if not trace_path.exists():
        return None
    n = 0
    for line in trace_path.read_text().splitlines():
        d = json.loads(line)
        for e in d["trace"]:
            if e.get("role") == "retrieval" and e.get("cache_miss_reason") == "ttl_expired":
                n += 1
    return n


def main() -> None:
    rows = []
    warnings = []

    for window, result_path, trace_fname in RUNS:
        if not result_path.exists():
            warnings.append(f"missing committed result: {result_path}")
            continue
        d = json.loads(result_path.read_text())
        s = d["summary"]
        git_sha = d.get("git_sha", "")

        ttl_lapses = count_ttl_lapses(RESULTS_DIR / trace_fname)
        if ttl_lapses is None:
            warnings.append(f"missing local trace for ttl_lapses: {trace_fname} (window {window})")

        # git_sha 6ba168e predates the enable_thinking=False fix (a714111+); flag
        # explicitly rather than silently presenting this as config-matched to
        # the other two windows, which run reasoning OFF.
        note = ""
        if git_sha.startswith("6ba168e"):
            note = "reasoning ON (pre-a714111) -- NOT config-matched to the other two windows"

        rows.append({
            "window": window,
            "t_prime": d.get("t_prime", ""),
            "T": d.get("t", ""),
            "freshness_errors": s["n_freshness_error"],
            "accuracy_pct": round(s["accuracy"] * 100, 1),
            "ttl_lapses": ttl_lapses if ttl_lapses is not None else "",
            "correct": s["n_correct"],
            "reasoning_errors": s["n_reasoning_error"],
            "parse_failures": s["n_parse_failure"],
            "n_tasks": d["n_tasks"],
            "git_sha": git_sha[:12],
            "reasoning_mode_note": note,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} rows written to {OUT.relative_to(PROJ)}")
    for r in rows:
        print(f"  {r['window']:<6} acc={r['accuracy_pct']}%  fe={r['freshness_errors']}  "
              f"ttl_lapses={r['ttl_lapses']}  sha={r['git_sha']}  {r['reasoning_mode_note']}")
    if warnings:
        print("\nWARNINGS:")
        for w_ in warnings:
            print(f"  {w_}")


if __name__ == "__main__":
    main()
