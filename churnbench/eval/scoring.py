"""Metric computation and verdict classification for the evaluation harness.

Four metrics per task:
    accuracy       — exact counts, 1% relative tolerance for USD, set-F1≥0.99 for lists
    cost_usd       — total LLM + embedding cost (embedding always $0 for local models)
    latency_s      — wall-clock seconds from arm.answer() call to result
    freshness_error — wrong at T AND right at T_eff; distinguished from reasoning_error

Verdict taxonomy (§4.5):
    correct          — answer matches ground truth at T
    freshness_error  — wrong at T but correct at T_eff (stale cache served old truth)
    reasoning_error  — wrong at both T and T_eff (model failure, not a staleness artifact)
    parse_failure    — LLM output could not be coerced to the task's answer_type

T_eff derivation (one rule per arm):
    naive          → T           (always live, no stale state possible — structural zero)
    classic_rag    → T_prime     (full index built at T_prime)
    hierarchical   → min(source_ts of each worker result; "live" maps to T)
    grounding      → min(last_refresh of staged retrievals; live routes map to T)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from churnbench.eval.parsing import parse
from churnbench.tasks.schema import GoldAnswer, Task


@dataclass
class TaskResult:
    """Fully-scored outcome of one arm's response to one task."""

    task_id: str
    arm: str
    task_tier: int
    task_intent: str
    question_text: str
    answer_raw: str
    answer_parsed: Any
    gold: Any  # gold value at T
    gold_at_t_eff: Any  # gold value at T_eff (for freshness attribution)
    verdict: str  # correct | freshness_error | reasoning_error | parse_failure
    t_eff: str  # ISO date of the effective retrieval time
    cost_usd: float
    latency_s: float
    input_tokens: int
    output_tokens: int
    embedding_tokens: int
    attribution: dict[str, Any]  # stale-artifact info for freshness_error grounding verdicts
    tool_calls: list[dict[str, Any]]


@dataclass
class SummaryMetrics:
    """Aggregate metrics across all tasks for one arm run."""

    n_tasks: int
    n_correct: int
    n_freshness_error: int
    n_reasoning_error: int
    n_parse_failure: int
    accuracy: float
    freshness_error_rate: float
    reasoning_error_rate: float
    parse_failure_rate: float
    mean_cost_usd: float
    total_cost_usd: float
    mean_latency_s: float
    mean_input_tokens: float
    mean_output_tokens: float


# ── Accuracy helpers ──────────────────────────────────────────────────────────


def is_correct(answer: Any, gold: Any, answer_type: str) -> bool:
    """Return True iff answer matches gold by the paper's per-type tolerance rules."""
    if answer is None or gold is None:
        return False
    if answer_type == "int":
        try:
            return int(answer) == int(gold)
        except (TypeError, ValueError):
            return False
    if answer_type == "float":
        try:
            a, g = float(answer), float(gold)
        except (TypeError, ValueError):
            return False
        if g == 0.0:
            return abs(a) < 1e-6
        return abs(a - g) / abs(g) <= 0.01
    if answer_type == "list[str]":
        try:
            ans_set = {str(v).strip().lower() for v in answer}
            gold_set = {str(v).strip().lower() for v in gold}
        except (TypeError, ValueError):
            return False
        if not ans_set and not gold_set:
            return True
        if not ans_set or not gold_set:
            return False
        intersection = len(ans_set & gold_set)
        precision = intersection / len(ans_set)
        recall = intersection / len(gold_set)
        denom = precision + recall
        if denom == 0.0:
            return False
        return (2.0 * precision * recall / denom) >= 0.99
    # str: case-insensitive strip match
    try:
        return str(answer).strip().lower() == str(gold).strip().lower()
    except (TypeError, ValueError):
        return False


# ── T_eff derivation ──────────────────────────────────────────────────────────


def compute_t_eff(
    arm_name: str,
    trace: list[dict[str, Any]],
    T_prime: date,
    T: date,
) -> date:
    """Return the effective retrieval time for this arm's answer.

    naive       → T   (always live; freshness errors are structurally impossible)
    classic_rag → T_prime  (entire index frozen at T_prime)
    hierarchical → min(each worker's source_ts; "live" maps to T)
    grounding    → min(last_refresh of staged retrievals; live routes map to T)
    """
    if arm_name == "naive":
        return T

    if arm_name == "classic_rag":
        return T_prime

    if arm_name == "hierarchical":
        ts_dates: list[date] = []
        for entry in trace:
            role = entry.get("role", "")
            if "_worker" in role:
                raw_ts = entry.get("source_ts", "live")
                if raw_ts == "live":
                    ts_dates.append(T)
                else:
                    try:
                        ts_dates.append(date.fromisoformat(str(raw_ts)))
                    except (ValueError, TypeError):
                        ts_dates.append(T)
        # No workers (e.g. direct_answer path) → conservative: T_prime
        return min(ts_dates) if ts_dates else T_prime

    if arm_name.startswith("grounding"):
        t_eff_dates: list[date] = []
        for entry in trace:
            if entry.get("role") != "retrieval":
                continue
            if entry.get("staged_vs_live") == "staged":
                lr = entry.get("last_refresh")
                if lr:
                    try:
                        t_eff_dates.append(date.fromisoformat(str(lr)))
                    except (ValueError, TypeError):
                        t_eff_dates.append(T)
                else:
                    t_eff_dates.append(T)
            else:
                # Live routes contribute T (inherently current)
                t_eff_dates.append(T)
        # No retrieval trace entries (e.g. no_source_routing docs-only path) → T_prime
        return min(t_eff_dates) if t_eff_dates else T_prime

    # Unknown arm name → conservative: T (no freshness errors credited)
    return T


# ── Verdict classification ────────────────────────────────────────────────────


def classify_verdict(
    answer_raw: str,
    answer_type: str,
    gold_at_T: GoldAnswer,
    gold_at_t_eff: GoldAnswer,
    t_eff: date,
    T: date,
) -> str:
    """Return the verdict string for one task result.

    Calls parse() internally so parse_failure is detected here, not in arm code.
    """
    pr = parse(answer_raw, answer_type)
    if pr.parse_failure:
        return "parse_failure"
    if is_correct(pr.value, gold_at_T.value, answer_type):
        return "correct"
    # Wrong at T: check if it was right at T_eff (only meaningful when T_eff < T)
    if t_eff < T and is_correct(pr.value, gold_at_t_eff.value, answer_type):
        return "freshness_error"
    return "reasoning_error"


# ── Freshness-error attribution ───────────────────────────────────────────────


def stale_artifact_attribution(
    arm_name: str,
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return stale-artifact metadata for grounding arm freshness errors.

    Names each staged retrieval that contributed stale data, enabling Table 4's
    'freshness attribution' column.
    """
    if not arm_name.startswith("grounding"):
        return {}
    stale: list[dict[str, Any]] = []
    for entry in trace:
        if entry.get("role") == "retrieval" and entry.get("staged_vs_live") == "staged":
            stale.append(
                {
                    "entity_class": entry.get("entity_class"),
                    "route": entry.get("route"),
                    "last_refresh": entry.get("last_refresh"),
                    "cache_miss_reason": entry.get("cache_miss_reason"),
                    "measure": entry.get("measure"),
                }
            )
    return {"stale_artifacts": stale} if stale else {}


# ── Summary aggregation ───────────────────────────────────────────────────────


def compute_summary(results: list[TaskResult]) -> SummaryMetrics:
    """Aggregate verdicts and costs across all task results."""
    n = len(results)
    if n == 0:
        return SummaryMetrics(
            n_tasks=0,
            n_correct=0,
            n_freshness_error=0,
            n_reasoning_error=0,
            n_parse_failure=0,
            accuracy=0.0,
            freshness_error_rate=0.0,
            reasoning_error_rate=0.0,
            parse_failure_rate=0.0,
            mean_cost_usd=0.0,
            total_cost_usd=0.0,
            mean_latency_s=0.0,
            mean_input_tokens=0.0,
            mean_output_tokens=0.0,
        )
    n_correct = sum(1 for r in results if r.verdict == "correct")
    n_fresh = sum(1 for r in results if r.verdict == "freshness_error")
    n_reason = sum(1 for r in results if r.verdict == "reasoning_error")
    n_parse = sum(1 for r in results if r.verdict == "parse_failure")
    total_cost = sum(r.cost_usd for r in results)
    return SummaryMetrics(
        n_tasks=n,
        n_correct=n_correct,
        n_freshness_error=n_fresh,
        n_reasoning_error=n_reason,
        n_parse_failure=n_parse,
        accuracy=n_correct / n,
        freshness_error_rate=n_fresh / n,
        reasoning_error_rate=n_reason / n,
        parse_failure_rate=n_parse / n,
        mean_cost_usd=total_cost / n,
        total_cost_usd=total_cost,
        mean_latency_s=sum(r.latency_s for r in results) / n,
        mean_input_tokens=sum(r.input_tokens for r in results) / n,
        mean_output_tokens=sum(r.output_tokens for r in results) / n,
    )


# ── Config hash ───────────────────────────────────────────────────────────────


def config_hash(
    arm_name: str,
    arm_flags: dict[str, Any],
    model: str,
    T_prime: date,
    T: date,
    seed: int,
    tasks: list[Task],
) -> str:
    """Deterministic 16-char hex hash identifying a unique experiment configuration."""
    task_ids = sorted(t.task_id for t in tasks)
    task_hash = hashlib.sha256(json.dumps(task_ids, sort_keys=True).encode()).hexdigest()[:16]
    payload = {
        "arm": arm_name,
        "flags": arm_flags,
        "model": model,
        "T_prime": T_prime.isoformat(),
        "T": T.isoformat(),
        "seed": seed,
        "task_hash": task_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# ── JSONL serialization helpers ───────────────────────────────────────────────


def task_result_to_dict(r: TaskResult) -> dict[str, Any]:
    d = asdict(r)
    return d


def task_result_from_dict(d: dict[str, Any]) -> TaskResult:
    return TaskResult(
        task_id=d["task_id"],
        arm=d["arm"],
        task_tier=int(d["task_tier"]),
        task_intent=str(d["task_intent"]),
        question_text=str(d["question_text"]),
        answer_raw=str(d["answer_raw"]),
        answer_parsed=d["answer_parsed"],
        gold=d["gold"],
        gold_at_t_eff=d["gold_at_t_eff"],
        verdict=str(d["verdict"]),
        t_eff=str(d["t_eff"]),
        cost_usd=float(d["cost_usd"]),
        latency_s=float(d["latency_s"]),
        input_tokens=int(d["input_tokens"]),
        output_tokens=int(d["output_tokens"]),
        embedding_tokens=int(d["embedding_tokens"]),
        attribution=dict(d.get("attribution", {})),
        tool_calls=list(d.get("tool_calls", [])),
    )
