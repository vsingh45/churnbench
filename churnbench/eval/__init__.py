"""ChurnBench: scoring and evaluation utilities."""

from churnbench.eval.harness import RunHarness
from churnbench.eval.parsing import ParseResult, parse
from churnbench.eval.report import generate_report, load_arm_results
from churnbench.eval.scoring import (
    SummaryMetrics,
    TaskResult,
    classify_verdict,
    compute_summary,
    compute_t_eff,
    config_hash,
    is_correct,
    stale_artifact_attribution,
)

__all__ = [
    "ParseResult",
    "RunHarness",
    "SummaryMetrics",
    "TaskResult",
    "classify_verdict",
    "compute_summary",
    "compute_t_eff",
    "config_hash",
    "generate_report",
    "is_correct",
    "load_arm_results",
    "parse",
    "stale_artifact_attribution",
]
