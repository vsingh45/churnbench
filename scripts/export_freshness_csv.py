"""Export complete freshness-error tables as CSV for the paper's supplementary material."""
import csv
import json
import sys
from pathlib import Path

PROJ = Path("/Users/vivekkumarsingh/Documents/All-Proj/churnbench")
RESULTS_DIR = PROJ / "data/full/results"
OUT_DIR = PROJ / "results/supplementary"
TASKS_FILE = PROJ / "data/full/tasks.jsonl"

sys.path.insert(0, str(PROJ))
from churnbench.arms.grounding.semantic_model import ENTITY_REGISTRY
from churnbench.eval.scoring import is_correct
from churnbench.tasks.schema import load_tasks as load_tasks_typed

# run_id -> config hash (or None if not yet run)
RUNS = {
    "D1_prefix_baseline":   "06e9fbdfe5d0b7d4.PREFIX_BASELINE_BACKUP",  # special-cased path below
    "D1_matched_baseline":  "06e9fbdfe5d0b7d4",
    "D14_baseline":         "31e6a01df61bf4d5",
    "D28_prefix_baseline":  "6326534d90587f84.PREFIX_BASELINE_BACKUP",  # special-cased path below
    "D28_matched_baseline": "6326534d90587f84",
    "D28_ablation":         "1d64ea6fe73b4de3",
    "D1_ablation":          "71e75e0867f37c01",
}
ARM_PREFIX = {
    "D1_prefix_baseline": "grounding", "D1_matched_baseline": "grounding", "D14_baseline": "grounding",
    "D28_prefix_baseline": "grounding", "D28_matched_baseline": "grounding",
    "D28_ablation": "grounding_no_freshness_tiers", "D1_ablation": "grounding_no_freshness_tiers",
}
WINDOW = {
    "D1_prefix_baseline": "D+1", "D1_matched_baseline": "D+1", "D14_baseline": "D+14",
    "D28_prefix_baseline": "D+28",
    "D28_matched_baseline": "D+28", "D28_ablation": "D+28", "D1_ablation": "D+1",
}

typed_tasks = {t.task_id: t for t in load_tasks_typed(TASKS_FILE)}

COLUMNS = [
    "run_id", "git_sha", "window", "t_prime", "t", "task_id", "resolver_ref",
    "entity_class", "freshness_tier", "ttl_days", "route", "query_method",
    "t_eff", "arm_answer", "gold_at_t_eff", "gold_at_t", "definitional_check_passed",
]


def load_run(run_id, cfg_hash):
    arm = ARM_PREFIX[run_id]
    json_path = RESULTS_DIR / f"{arm}__{cfg_hash}.json"
    trace_path = RESULTS_DIR / f"{arm}__{cfg_hash}.traces.jsonl"
    if not json_path.exists():
        return None, None
    d = json.loads(json_path.read_text())
    traces = {}
    if trace_path.exists():
        for line in trace_path.read_text().splitlines():
            t = json.loads(line)
            traces[t["task_id"]] = t["trace"]
    return d, traces


def rows_for_run(run_id, cfg_hash):
    d, traces = load_run(run_id, cfg_hash)
    if d is None:
        return None  # not yet available
    rows = []
    for r in d["results"]:
        if r["verdict"] != "freshness_error":
            continue
        tid = r["task_id"]
        task = typed_tasks[tid]
        tr = traces.get(tid, [])
        retrievals = [e for e in tr if e.get("role") == "retrieval"]
        route = ",".join(e.get("route", "") for e in retrievals)
        query_method = ",".join(e.get("query_method", "") for e in retrievals)
        ec_name = retrievals[0].get("entity_class") if retrievals else ""
        ec = ENTITY_REGISTRY.get(ec_name)
        ttl_days = ec.ttl_days if ec else None

        answer = r["answer_parsed"]
        gold_teff = r["gold_at_t_eff"]
        gold_t = r["gold"]
        passed = is_correct(answer, gold_teff, task.answer_type) and not is_correct(
            answer, gold_t, task.answer_type
        )

        rows.append(
            {
                "run_id": run_id,
                "git_sha": d.get("git_sha", ""),
                "window": WINDOW[run_id],
                "t_prime": d.get("t_prime", ""),
                "t": d.get("t", ""),
                "task_id": tid,
                "resolver_ref": task.resolver_ref,
                "entity_class": ec_name,
                "freshness_tier": task.tier,
                "ttl_days": ttl_days,
                "route": route,
                "query_method": query_method,
                "t_eff": r["t_eff"],
                "arm_answer": answer,
                "gold_at_t_eff": gold_teff,
                "gold_at_t": gold_t,
                "definitional_check_passed": passed,
            }
        )
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    status = {}
    for run_id, cfg_hash in RUNS.items():
        rows = rows_for_run(run_id, cfg_hash)
        if rows is None:
            status[run_id] = "PENDING (not yet run)"
            continue
        status[run_id] = f"{len(rows)} freshness errors"
        out_path = OUT_DIR / f"freshness_errors_{run_id}.csv"
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        all_rows.extend(rows)

    combined_path = OUT_DIR / "freshness_errors_ALL_RUNS.csv"
    with combined_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(all_rows)

    print("=== Export status ===")
    for run_id, s in status.items():
        print(f"  {run_id}: {s}")
    print(f"\nCombined CSV: {combined_path} ({len(all_rows)} total rows)")
    n_pending = sum(1 for s in status.values() if s.startswith("PENDING"))
    print(f"\n{n_pending} run(s) still pending — re-run this script after they complete.")


if __name__ == "__main__":
    main()
