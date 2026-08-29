"""Analyze grounding sweep results: per-tier, per-route accuracy, verdict breakdown."""
import json
from pathlib import Path
from collections import defaultdict

PROJ = Path("/Users/vivekkumarsingh/Documents/All-Proj/churnbench")
RESULTS_DIR = PROJ / "data/full/results"
TASKS_FILE = PROJ / "data/full/tasks.jsonl"

# Design-B: evaluation T fixed at 2024-04-27 (all task.T values); T_prime varies.
# D+1  = cache 1 day old  (T_prime = T - 1)
# D+14 = cache 14 days old (T_prime = T - 14)
# D+28 = cache 28 days old (T_prime = T - 28)
WINDOWS = [
    ("D+1",  "2024-04-26", "2024-04-27"),
    ("D+14", "2024-04-13", "2024-04-27"),
    ("D+28", "2024-03-29", "2024-04-27"),
]


def _load_task_meta() -> dict[str, dict]:
    """Return {task_id: {resolver_ref, tier}} from tasks.jsonl."""
    meta: dict[str, dict] = {}
    if TASKS_FILE.exists():
        for line in TASKS_FILE.read_text().strip().splitlines():
            t = json.loads(line)
            meta[t["task_id"]] = {"resolver_ref": t.get("resolver_ref", "unknown"), "tier": t.get("tier", 0)}
    return meta


def find_result_file(t_prime: str, t: str) -> Path | None:
    """Find the result JSON whose config matches t_prime and t."""
    for p in sorted(RESULTS_DIR.glob("grounding__*.json")):
        d = json.loads(p.read_text())
        if d.get("t_prime") == t_prime and d.get("t") == t:
            return p
    return None


def analyze(data: dict, task_meta: dict[str, dict]) -> dict:
    results = data["results"]

    # Overall
    verdicts: dict[str, int] = defaultdict(int)
    for r in results:
        verdicts[r["verdict"]] += 1

    n = len(results)
    correct = verdicts["correct"]
    accuracy = correct / n if n else 0

    # Per-tier (use task_tier from result, fall back to tasks.jsonl)
    tier_stats: dict[int, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        tier = r.get("task_tier") or task_meta.get(r["task_id"], {}).get("tier", 0)
        tier_stats[tier]["total"] += 1
        if r["verdict"] == "correct":
            tier_stats[tier]["correct"] += 1

    per_tier = {
        tier: {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": s["correct"] / s["total"] if s["total"] else 0,
        }
        for tier, s in sorted(tier_stats.items())
    }

    # Per-route (resolver_ref from tasks.jsonl)
    route_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0, "freshness": 0, "reasoning": 0, "parse": 0})
    for r in results:
        route = task_meta.get(r["task_id"], {}).get("resolver_ref", "unknown")
        route_stats[route]["total"] += 1
        v = r["verdict"]
        if v == "correct":
            route_stats[route]["correct"] += 1
        elif v == "freshness_error":
            route_stats[route]["freshness"] += 1
        elif v == "reasoning_error":
            route_stats[route]["reasoning"] += 1
        elif v == "parse_failure":
            route_stats[route]["parse"] += 1

    per_route = {
        route: {
            "accuracy": s["correct"] / s["total"] if s["total"] else 0,
            "n": s["total"],
            "freshness": s["freshness"],
            "reasoning": s["reasoning"],
            "parse": s["parse"],
        }
        for route, s in sorted(route_stats.items())
    }

    return {
        "n": n,
        "accuracy": accuracy,
        "correct": correct,
        "verdicts": dict(verdicts),
        "per_tier": per_tier,
        "per_route": per_route,
    }


def print_window(label: str, t_prime: str, t: str, task_meta: dict[str, dict]) -> dict | None:
    p = find_result_file(t_prime, t)
    if p is None:
        print(f"\n{'='*60}")
        print(f"{label} (T'={t_prime}, T={t}): NO RESULT FILE YET")
        return None
    data = json.loads(p.read_text())
    a = analyze(data, task_meta)

    print(f"\n{'='*60}")
    print(f"{label}  T'={t_prime}  T={t}  (config={p.stem.split('__')[1]})")
    print(f"Accuracy: {a['correct']}/{a['n']} = {a['accuracy']:.1%}")
    v = a["verdicts"]
    print(f"Verdicts: correct={v.get('correct',0)}  freshness_error={v.get('freshness_error',0)}  reasoning_error={v.get('reasoning_error',0)}  parse_failure={v.get('parse_failure',0)}")

    print("\nPer-tier:")
    for tier, s in a["per_tier"].items():
        print(f"  tier-{tier}: {s['correct']}/{s['total']} = {s['accuracy']:.1%}")

    print("\nPer-route (sorted by accuracy asc):")
    routes_sorted = sorted(a["per_route"].items(), key=lambda x: x[1]["accuracy"])
    for route, s in routes_sorted:
        fe = f" fe={s['freshness']}" if s["freshness"] else ""
        re = f" re={s['reasoning']}" if s["reasoning"] else ""
        pf = f" pf={s['parse']}" if s["parse"] else ""
        print(f"  {route:<40} {s['accuracy']:.1%}  (n={s['n']}{fe}{re}{pf})")
    return a


def main() -> None:
    task_meta = _load_task_meta()
    results = {}
    for label, t_prime, t in WINDOWS:
        r = print_window(label, t_prime, t, task_meta)
        if r:
            results[label] = r

    # Headline
    fe_counts = [(lbl, r["verdicts"].get("freshness_error", 0)) for lbl, r in results.items()]
    if len(fe_counts) >= 2:
        print(f"\n{'='*60}")
        print("HEADLINE — freshness_error growth:")
        for lbl, fe in fe_counts:
            print(f"  {lbl}: {fe}")
        if len(fe_counts) == 3:
            d1, d14, d28 = [v for _, v in fe_counts]
            print(f"  Trend: {d1} → {d14} → {d28}  (predictable={d1 <= d14 <= d28})")


if __name__ == "__main__":
    main()
