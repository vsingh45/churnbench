"""Compute task-set composition statistics cited in the paper, from the
committed task set and the semantic model registry -- never hardcoded.
Writes results/supplementary/task_set_stats.json.

Run: poetry run python3 scripts/compute_task_set_stats.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from churnbench.arms.grounding.semantic_model import MEASURE_TO_ENTITY  # noqa: E402
from churnbench.tasks.schema import load_tasks  # noqa: E402

TASKS_FILE = PROJ / "data/full/tasks.jsonl"
OUT = PROJ / "results/supplementary/task_set_stats.json"


def trivial_zero_score(tasks) -> tuple[int, int, float]:
    """An always-answer-zero/empty arm's score against gold@T for each task.
    Uses the ledger + resolver to compute real gold values -- same method as
    tests/test_tasks.py::test_trivial_zero_arm_score_below_10pct.
    """
    from churnbench.ledger.ledger import Ledger
    from churnbench.tasks.resolver import LedgerResolver

    ledger = Ledger.load(PROJ / "data/full/ledger.jsonl")
    resolver = LedgerResolver(ledger)

    trivial_correct = 0
    for t in tasks:
        gold = t.gold(resolver, at=t.T).value
        if t.answer_type == "int" and gold == 0:
            trivial_correct += 1
        elif t.answer_type == "float" and gold == 0.0:
            trivial_correct += 1
        elif t.answer_type == "str" and gold == "":
            trivial_correct += 1
        elif t.answer_type == "list[str]" and gold == []:
            trivial_correct += 1
    return trivial_correct, len(tasks), trivial_correct / len(tasks)


def main() -> None:
    tasks = load_tasks(TASKS_FILE)
    n_total = len(tasks)

    tier_counts = Counter(t.tier for t in tasks)
    intent_counts = Counter(t.intent for t in tasks)
    measure_counts = Counter(t.resolver_ref for t in tasks)

    registered_measures = set(MEASURE_TO_ENTITY.keys())
    all_measures = set(measure_counts.keys())
    registered_present = all_measures & registered_measures
    unregistered_present = all_measures - registered_measures

    tasks_registered = sum(c for m, c in measure_counts.items() if m in registered_measures)
    tasks_unregistered = sum(c for m, c in measure_counts.items() if m not in registered_measures)

    trivial_correct, trivial_n, trivial_score = trivial_zero_score(tasks)

    data = {
        "computed_by": "scripts/compute_task_set_stats.py",
        "source_files": ["data/full/tasks.jsonl", "data/full/ledger.jsonl",
                          "churnbench/arms/grounding/semantic_model.py"],
        "total_tasks": n_total,
        "tier_counts": {str(k): v for k, v in sorted(tier_counts.items())},
        "intent_counts": dict(sorted(intent_counts.items(), key=lambda kv: -kv[1])),
        "distinct_measures": len(measure_counts),
        "measures_registered_in_semantic_model": len(registered_present),
        "measures_not_registered": len(unregistered_present),
        "registered_measure_names": sorted(registered_present),
        "unregistered_measure_names": sorted(unregistered_present),
        "tasks_targeting_registered_measures": tasks_registered,
        "tasks_targeting_unregistered_measures": tasks_unregistered,
        "trivial_zero_arm": {
            "correct": trivial_correct,
            "n_tasks": trivial_n,
            "score": round(trivial_score, 4),
            "score_pct": round(trivial_score * 100, 1),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2) + "\n")

    print(f"total_tasks={n_total}")
    print(f"tier_counts={dict(tier_counts)}")
    print(f"distinct_measures={len(measure_counts)}  registered={len(registered_present)}  unregistered={len(unregistered_present)}")
    print(f"tasks_registered={tasks_registered}  tasks_unregistered={tasks_unregistered}")
    print(f"trivial_zero_arm: {trivial_correct}/{trivial_n} = {trivial_score:.1%}")
    print(f"written: {OUT.relative_to(PROJ)}")


if __name__ == "__main__":
    main()
