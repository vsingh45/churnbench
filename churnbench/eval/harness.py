"""Evaluation harness — orchestrates the full experiment lifecycle.

Six-step lifecycle (in order):
    1. Project at T_prime — freeze the fabric to the cache-build time
    2. arm.setup()       — build indexes and staged caches at T_prime
    3. Refresh scheduling (grounding arm only):
           Walk day-by-day from T_prime+1 to T.  For each day, call
           refresh_due() on the arm's registry; if any entity classes are
           stale, re-project the fabric to that day's state and re-run ETL
           for each stale entity, stamping last_refresh=day.
           This is the only place the tiered-staleness lifecycle is exercised
           against real projected data — the paper's §6.3 headline claim
           depends on this being correct.
    4. Project at T      — bring the fabric to the evaluation timestamp
    5. Run arm.answer()  — checkpoint every 10 tasks for resumability
    6. Score + write     — per-task verdict, config-hash-named JSON output

Results layout:
    results/<run>/<arm>__<cfg_hash>.json       summary + per-task (committed)
    results/<run>/<arm>__<cfg_hash>.traces.jsonl  full traces (gitignored)
    results/<run>/<arm>__<cfg_hash>.ckpt.json  checkpoint (deleted on success)
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

from churnbench.arms.base import ArmResult, BaseArm, FabricConfig
from churnbench.arms.grounding import GroundingArm
from churnbench.arms.grounding.etl import refresh_due, refresh_entity
from churnbench.eval.scoring import (
    SummaryMetrics,
    TaskResult,
    classify_verdict,
    compute_summary,
    compute_t_eff,
    config_hash,
    stale_artifact_attribution,
    task_result_from_dict,
    task_result_to_dict,
)
from churnbench.ledger.ledger import Ledger
from churnbench.tasks.resolver import LedgerResolver
from churnbench.tasks.schema import Task


# Injectable projector type: (ledger, T, docs_dir) → None
ProjectorFn = Callable[[Ledger, date, Path], None]

_CHECKPOINT_EVERY = 10


def _default_projector(ledger: Ledger, T: date, docs_dir: Path) -> None:
    """Production projector: wipe-and-rebuild Postgres, Mongo, docs at T."""
    from churnbench.fabric.projector import Projector

    proj = Projector()
    proj.project_postgres(ledger, T)
    proj.project_mongo(ledger, T)
    proj.project_docs(ledger, T, docs_dir)


class RunHarness:
    """Orchestrates one arm's complete lifecycle for one experiment run."""

    def __init__(
        self,
        run_dir: Path,
        fabric_config: FabricConfig,
        *,
        _projector: ProjectorFn | None = None,
    ) -> None:
        self._run_dir = run_dir
        self._config = fabric_config
        self._project: ProjectorFn = _projector or _default_projector
        run_dir.mkdir(parents=True, exist_ok=True)

    # ── Fabric projection ─────────────────────────────────────────────────────

    def _project_at(self, ledger: Ledger, T: date) -> None:
        self._project(ledger, T, self._config.docs_dir)

    # ── Refresh scheduling ────────────────────────────────────────────────────

    def _schedule_refreshes(
        self, arm: BaseArm, ledger: Ledger, T_prime: date, T: date
    ) -> None:
        """Walk T_prime+1 .. T and run ETL for each entity that goes stale.

        Only the grounding arm has a managed registry; other arms are no-ops.
        For each day that any entity's TTL lapses:
          1. Re-project the fabric to that day's state (so ETL reads current data).
          2. For every stale entity: run refresh_entity + stamp last_refresh.
        This is the only place the tiered-staleness lifecycle is exercised end-to-end.
        """
        if not isinstance(arm, GroundingArm):
            return

        day = T_prime + timedelta(days=1)
        while day <= T:
            due = refresh_due(arm._registry, day)
            if due:
                self._project_at(ledger, day)
                for ec in due:
                    refresh_entity(
                        ec.name,
                        arm._staged_engine,
                        pg_engine=arm._pg_engine,
                        mongo_db=arm._mongo_db,
                        T_prime=day,
                    )
                    ec.last_refresh = day
            day += timedelta(days=1)

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _ckpt_path(self, arm_name: str, cfg_hash: str) -> Path:
        return self._run_dir / f"{arm_name}__{cfg_hash}.ckpt.json"

    def _result_path(self, arm_name: str, cfg_hash: str) -> Path:
        return self._run_dir / f"{arm_name}__{cfg_hash}.json"

    def _trace_path(self, arm_name: str, cfg_hash: str) -> Path:
        return self._run_dir / f"{arm_name}__{cfg_hash}.traces.jsonl"

    def _load_checkpoint(
        self, path: Path
    ) -> tuple[set[str], list[TaskResult]]:
        """Return (completed_task_ids, completed_results) from a checkpoint file."""
        if not path.exists():
            return set(), []
        try:
            data = json.loads(path.read_text())
            results = [task_result_from_dict(r) for r in data.get("results", [])]
            return {r.task_id for r in results}, results
        except Exception:
            return set(), []

    def _write_checkpoint(self, path: Path, results: list[TaskResult]) -> None:
        path.write_text(
            json.dumps({"results": [task_result_to_dict(r) for r in results]})
        )

    def _write_results(
        self,
        path: Path,
        arm_name: str,
        cfg_hash: str,
        results: list[TaskResult],
        summary: SummaryMetrics,
    ) -> None:
        from dataclasses import asdict

        data: dict[str, Any] = {
            "arm": arm_name,
            "config_hash": cfg_hash,
            "n_tasks": len(results),
            "summary": asdict(summary),
            "results": [task_result_to_dict(r) for r in results],
        }
        path.write_text(json.dumps(data, indent=2))

    # ── Full lifecycle ────────────────────────────────────────────────────────

    def run(
        self,
        arm: BaseArm,
        arm_name: str,
        T_prime: date,
        T: date,
        ledger: Ledger,
        tasks: list[Task],
        resolver: LedgerResolver,
        *,
        model: str | None = None,
        seed: int = 42,
        arm_flags: dict[str, Any] | None = None,
    ) -> list[TaskResult]:
        """Execute the six-step lifecycle and return all TaskResults.

        Resumable: if a checkpoint exists from a prior interrupted run with the
        same config_hash, completed tasks are skipped and not re-sent to the arm.
        """
        _model = model or os.environ.get("CHURNBENCH_MODEL", "claude-sonnet-4-6")
        flags = arm_flags or {}
        cfg_hash = config_hash(arm_name, flags, _model, T_prime, T, seed, tasks)

        result_path = self._result_path(arm_name, cfg_hash)
        ckpt_path = self._ckpt_path(arm_name, cfg_hash)
        trace_path = self._trace_path(arm_name, cfg_hash)

        # Step 1 — project at T_prime
        self._project_at(ledger, T_prime)

        # Step 2 — arm.setup at T_prime
        arm.setup(self._config, T_prime)

        # Step 3 — refresh scheduling (grounding arm only)
        self._schedule_refreshes(arm, ledger, T_prime, T)

        # Step 4 — project at T for arm.answer
        self._project_at(ledger, T)

        # Step 5 — load checkpoint (resume)
        completed_ids, all_results = self._load_checkpoint(ckpt_path)
        pending = [t for t in tasks if t.task_id not in completed_ids]

        # Step 6 — run arm.answer for each pending task
        for i, task in enumerate(pending):
            arm_result: ArmResult = arm.answer(task)

            t_eff = compute_t_eff(arm_name, arm_result.trace, T_prime, T)
            gold_at_T = task.gold(resolver, at=T)
            gold_at_t_eff = task.gold(resolver, at=t_eff)

            verdict = classify_verdict(
                arm_result.answer_raw,
                task.answer_type,
                gold_at_T,
                gold_at_t_eff,
                t_eff,
                T,
            )

            attr: dict[str, Any] = {}
            if verdict == "freshness_error":
                attr = stale_artifact_attribution(arm_name, arm_result.trace)

            tr = TaskResult(
                task_id=task.task_id,
                arm=arm_name,
                task_tier=task.tier,
                task_intent=task.intent,
                question_text=task.question_text,
                answer_raw=arm_result.answer_raw,
                answer_parsed=arm_result.answer_parsed,
                gold=gold_at_T.value,
                gold_at_t_eff=gold_at_t_eff.value,
                verdict=verdict,
                t_eff=t_eff.isoformat(),
                cost_usd=arm_result.cost_usd,
                latency_s=arm_result.latency_s,
                input_tokens=arm_result.input_tokens,
                output_tokens=arm_result.output_tokens,
                embedding_tokens=arm_result.embedding_tokens,
                attribution=attr,
                tool_calls=arm_result.tool_calls,
            )
            all_results.append(tr)

            # Append trace to JSONL (gitignored raw traces)
            with trace_path.open("a") as tf:
                tf.write(
                    json.dumps(
                        {"task_id": task.task_id, "trace": arm_result.trace}
                    )
                    + "\n"
                )

            # Checkpoint every N tasks
            if (i + 1) % _CHECKPOINT_EVERY == 0:
                self._write_checkpoint(ckpt_path, all_results)

        # Step 7 — write final results + clean up checkpoint
        summary = compute_summary(all_results)
        self._write_results(result_path, arm_name, cfg_hash, all_results, summary)
        ckpt_path.unlink(missing_ok=True)

        arm.teardown()
        return all_results
