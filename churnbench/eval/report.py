"""Generate the paper's result tables as Markdown.

Sections:
    1. Arms × metrics table (Table 3): accuracy, freshness-error rate,
       reasoning-error rate, mean cost, mean latency — pooled and by tier.
    2. Ablation deltas (Table 4): grounding vs. each ablated variant.
    3. Cost decomposition: LLM input/output token counts and dollar totals.
    4. Freshness attribution (grounding arm only): which entity classes
       contributed to freshness errors and how often.

All [DATA REQUIRED] placeholders are filled exclusively from results/*.json
written by RunHarness.run() — never from hardcoded numbers.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from churnbench.eval.scoring import SummaryMetrics, TaskResult, compute_summary


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _fmt_cost(v: float) -> str:
    return f"${v:.6f}"


def _fmt_latency(v: float) -> str:
    return f"{v:.2f}s"


# ── Table builders ────────────────────────────────────────────────────────────


def _summary_row(arm: str, m: SummaryMetrics) -> str:
    return (
        f"| {arm:<30} | {m.n_tasks:>6} "
        f"| {_pct(m.accuracy):>8} "
        f"| {_pct(m.freshness_error_rate):>8} "
        f"| {_pct(m.reasoning_error_rate):>8} "
        f"| {_pct(m.parse_failure_rate):>8} "
        f"| {_fmt_cost(m.mean_cost_usd):>12} "
        f"| {_fmt_latency(m.mean_latency_s):>9} |"
    )


def arms_metrics_table(results_by_arm: dict[str, list[TaskResult]]) -> str:
    """Build Table 3: arms × metrics (pooled)."""
    header = (
        "| Arm                           | Tasks "
        "| Accuracy | Fresh-err | Reason-err | Parse-fail "
        "| Mean Cost    | Mean Latency |"
    )
    sep = (
        "|-------------------------------|-------"
        "|----------|-----------|------------|------------"
        "|--------------|--------------|"
    )
    rows = [header, sep]
    for arm_name, task_results in sorted(results_by_arm.items()):
        m = compute_summary(task_results)
        rows.append(_summary_row(arm_name, m))
    return "\n".join(rows)


def arms_metrics_by_tier_table(results_by_arm: dict[str, list[TaskResult]]) -> str:
    """Build per-tier breakdown of Table 3."""
    lines: list[str] = []
    for arm_name, task_results in sorted(results_by_arm.items()):
        tier_groups: dict[int, list[TaskResult]] = defaultdict(list)
        for r in task_results:
            tier_groups[r.task_tier].append(r)
        if not tier_groups:
            continue
        lines.append(f"\n### {arm_name}")
        header = (
            "| Tier | Tasks "
            "| Accuracy | Fresh-err | Reason-err | Mean Cost    | Mean Latency |"
        )
        sep = (
            "|------|-------"
            "|----------|-----------|------------|--------------|--------------|"
        )
        lines.extend([header, sep])
        for tier in sorted(tier_groups.keys()):
            m = compute_summary(tier_groups[tier])
            lines.append(
                f"| {tier:>4} | {m.n_tasks:>5} "
                f"| {_pct(m.accuracy):>8} "
                f"| {_pct(m.freshness_error_rate):>9} "
                f"| {_pct(m.reasoning_error_rate):>10} "
                f"| {_fmt_cost(m.mean_cost_usd):>12} "
                f"| {_fmt_latency(m.mean_latency_s):>12} |"
            )
    return "\n".join(lines)


def ablation_delta_table(results_by_arm: dict[str, list[TaskResult]]) -> str:
    """Build Table 4: delta from full grounding to each ablation."""
    baseline_results = results_by_arm.get("grounding", [])
    if not baseline_results:
        return "*(grounding arm results not found — cannot compute ablation deltas)*"

    baseline = compute_summary(baseline_results)

    ablations = [
        ("grounding_no_freshness_tiers", "No freshness tiers"),
        ("grounding_no_semantic_model", "No semantic model"),
        ("grounding_no_source_routing", "No source routing"),
    ]

    header = (
        "| Ablation           | ΔAccuracy | ΔFresh-err | ΔReason-err | ΔMean Cost |"
    )
    sep = "|--------------------|-----------|------------|-------------|------------|"
    rows = [header, sep]

    for arm_key, label in ablations:
        abl_results = results_by_arm.get(arm_key)
        if not abl_results:
            rows.append(f"| {label:<18} | *n/a*     | *n/a*      | *n/a*       | *n/a*      |")
            continue
        m = compute_summary(abl_results)
        d_acc = m.accuracy - baseline.accuracy
        d_fresh = m.freshness_error_rate - baseline.freshness_error_rate
        d_reason = m.reasoning_error_rate - baseline.reasoning_error_rate
        d_cost = m.mean_cost_usd - baseline.mean_cost_usd
        sign = lambda v: f"+{_pct(v)}" if v >= 0 else _pct(v)  # noqa: E731
        rows.append(
            f"| {label:<18} "
            f"| {sign(d_acc):>9} "
            f"| {sign(d_fresh):>10} "
            f"| {sign(d_reason):>11} "
            f"| {'+' if d_cost >= 0 else ''}{_fmt_cost(d_cost):>10} |"
        )

    return "\n".join(rows)


def cost_decomposition_table(results_by_arm: dict[str, list[TaskResult]]) -> str:
    """Show token-level cost breakdown: LLM input, LLM output, embedding, total."""
    header = (
        "| Arm                           | Tasks "
        "| Σ Input Tok | Σ Output Tok | Σ Embed Tok "
        "| Total $USD   |"
    )
    sep = (
        "|-------------------------------|-------"
        "|-------------|--------------|-------------"
        "|--------------|"
    )
    rows = [header, sep]
    for arm_name, task_results in sorted(results_by_arm.items()):
        n = len(task_results)
        if n == 0:
            continue
        tot_inp = sum(r.input_tokens for r in task_results)
        tot_out = sum(r.output_tokens for r in task_results)
        tot_emb = sum(r.embedding_tokens for r in task_results)
        tot_cost = sum(r.cost_usd for r in task_results)
        rows.append(
            f"| {arm_name:<30} | {n:>6} "
            f"| {tot_inp:>11,} "
            f"| {tot_out:>12,} "
            f"| {tot_emb:>11,} "
            f"| {_fmt_cost(tot_cost):>12} |"
        )
    return "\n".join(rows)


def freshness_attribution_table(results_by_arm: dict[str, list[TaskResult]]) -> str:
    """Count freshness errors by entity class for the grounding arm."""
    grounding_results = results_by_arm.get("grounding", [])
    fresh_errors = [r for r in grounding_results if r.verdict == "freshness_error"]
    if not fresh_errors:
        return "*(no freshness errors in grounding arm results)*"

    # Count by entity class
    ec_counts: dict[str, int] = defaultdict(int)
    for r in fresh_errors:
        for art in r.attribution.get("stale_artifacts", []):
            ec = art.get("entity_class") or "unknown"
            ec_counts[ec] += 1

    if not ec_counts:
        return "*(freshness errors found but no stale artifacts recorded)*"

    header = "| Entity Class          | Freshness Errors |"
    sep = "|-----------------------|-----------------|"
    rows = [header, sep]
    for ec, cnt in sorted(ec_counts.items(), key=lambda x: -x[1]):
        rows.append(f"| {ec:<21} | {cnt:>15} |")
    return "\n".join(rows)


# ── Full report ───────────────────────────────────────────────────────────────


def generate_report(
    results_by_arm: dict[str, list[TaskResult]],
    *,
    T_prime: str,
    T: str,
    out_path: Path,
) -> None:
    """Write a Markdown report to out_path.

    results_by_arm maps arm_name → list of TaskResult.
    T_prime, T are ISO date strings for the report header.
    """
    sections: list[str] = []

    sections.append("# ChurnBench Evaluation Report")
    sections.append(f"\n**Fabric frozen at:** T′ = {T_prime}  |  **Evaluated at:** T = {T}\n")

    sections.append("## Table 3 — Arms × Metrics (Pooled)\n")
    sections.append(arms_metrics_table(results_by_arm))

    sections.append("\n\n## Table 3b — Arms × Metrics by Tier\n")
    sections.append(arms_metrics_by_tier_table(results_by_arm))

    sections.append("\n\n## Table 4 — Ablation Deltas (vs. Full Grounding)\n")
    sections.append(ablation_delta_table(results_by_arm))

    sections.append("\n\n## Cost Decomposition\n")
    sections.append(cost_decomposition_table(results_by_arm))

    sections.append("\n\n## Freshness Attribution (Grounding Arm)\n")
    sections.append(freshness_attribution_table(results_by_arm))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections) + "\n")


# ── Load results helpers ──────────────────────────────────────────────────────


def load_arm_results(results_dir: Path) -> dict[str, list[TaskResult]]:
    """Load all <arm>__<hash>.json files from a results directory."""
    import json

    from churnbench.eval.scoring import task_result_from_dict

    by_arm: dict[str, list[TaskResult]] = {}
    for p in sorted(results_dir.glob("*.json")):
        # Skip checkpoints and summary files
        if ".ckpt." in p.name:
            continue
        try:
            data = json.loads(p.read_text())
            arm_name = str(data.get("arm", p.stem.split("__")[0]))
            task_results = [
                task_result_from_dict(r) for r in data.get("results", [])
            ]
            by_arm.setdefault(arm_name, []).extend(task_results)
        except Exception:
            continue
    return by_arm
