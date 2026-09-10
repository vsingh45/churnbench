"""Export one row per experimental run (Table -- run provenance) from every
committed results/*_full.json file. Makes the whole experimental matrix,
including pre-retry contaminated states, auditable in one file.

reasoning_mode is derived by checking git ancestry of each run's recorded
git_sha against the enable_thinking=False fix commit (a714111...), not
hardcoded per file. model is imported from churnbench.arms.base, the actual
source of truth, not restated.

Run: poetry run python3 scripts/export_run_provenance.py
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from churnbench.arms.base import _DEFAULT_MODEL  # noqa: E402

RESULTS_DIR = PROJ / "results"
OUT = RESULTS_DIR / "supplementary/run_provenance.csv"

ENABLE_THINKING_FIX_SHA = "a71411139e361929d33ff73820e94aff30a679f4"

COLUMNS = ["run_id", "config_hash", "git_sha", "arm", "ablation_flag", "t_prime", "T",
           "model", "reasoning_mode", "n_tasks", "permanent_failures_before_retry",
           "retried", "freshness_errors", "accuracy_pct"]


def reasoning_mode_for(git_sha: str) -> str:
    if not git_sha:
        return "unknown"
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ENABLE_THINKING_FIX_SHA, git_sha],
            cwd=PROJ, check=True, capture_output=True,
        )
        return "OFF"  # fix commit is an ancestor of this run's sha -> reasoning disabled
    except subprocess.CalledProcessError:
        return "ON"


def count_stub_failures(results: list[dict]) -> int:
    return sum(1 for r in results if r.get("answer_raw") == "reasoning_error")


def main() -> None:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*_full.json")):
        d = json.loads(path.read_text())
        run_id = path.stem.replace("_full", "")
        git_sha = d.get("git_sha", "")
        rp = d.get("retry_provenance")

        if rp is not None:
            # A targeted-retry pass was actually applied to this file.
            permanent_failures_before_retry = len(rp.get("retried_task_ids", []))
            retried = True
        else:
            # No retry_provenance -- either a pre-repair backup snapshot (count its
            # own stubs), or a run that used the hardened retry policy from the
            # start and needed no separate repair pass (0 stubs, retried=False).
            permanent_failures_before_retry = count_stub_failures(d["results"])
            retried = False

        s = d["summary"]
        rows.append({
            "run_id": run_id,
            "config_hash": d.get("config_hash", ""),
            "git_sha": git_sha[:12],
            "arm": d.get("arm", ""),
            "ablation_flag": d.get("arm", "") == "grounding_no_freshness_tiers",
            "t_prime": d.get("t_prime", ""),
            "T": d.get("t", ""),
            "model": _DEFAULT_MODEL,
            "reasoning_mode": reasoning_mode_for(git_sha),
            "n_tasks": d.get("n_tasks", ""),
            "permanent_failures_before_retry": permanent_failures_before_retry,
            "retried": retried,
            "freshness_errors": s["n_freshness_error"],
            "accuracy_pct": round(s["accuracy"] * 100, 1),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} rows written to {OUT.relative_to(PROJ)}")
    for r in rows:
        print(f"  {r['run_id']:<40} pf_before={r['permanent_failures_before_retry']:>2} "
              f"retried={r['retried']!s:<5} reasoning={r['reasoning_mode']:<3} "
              f"fe={r['freshness_errors']:>2} acc={r['accuracy_pct']}%")

    n_affected = sum(1 for r in rows if r["permanent_failures_before_retry"] > 0)
    print(f"\n{n_affected} of {len(rows)} rows show permanent failures at some point "
          f"(pre-retry state or unretried run).")


if __name__ == "__main__":
    main()
