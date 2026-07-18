"""Tests for the evaluation harness (churnbench/eval/).

Six test classes:

1. TestLifecycle        — critical integration: real refresh schedule against
                          the entity registry, with mock projector and mock LLM.
                          This is the first time the tiered-staleness machinery
                          runs against a real refresh schedule (not helper-stamped
                          timestamps) — validates the §6.3 headline mechanism.

2. TestTeff             — T_eff derivation per arm, from synthetic trace fixtures.

3. TestVerdicts         — verdict classification (correct, freshness_error,
                          reasoning_error, parse_failure) from raw answer strings.

4. TestNaiveValidity    — structural guarantee: naive arm cannot produce freshness
                          errors because T_eff = T by definition.

5. TestCheckpointResume — checkpoint file is loaded correctly; completed tasks are
                          returned without calling arm.answer().

6. TestConfigHash       — config_hash is stable under same args, differs under
                          any change to arm name, T, T_prime, seed, or task set.
"""

from __future__ import annotations

import copy
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from sqlalchemy import create_engine

from churnbench.arms.base import FabricConfig
from churnbench.arms.grounding import GroundingArm
from churnbench.arms.grounding.etl import create_staged_schema
from churnbench.arms.grounding.semantic_model import ENTITY_REGISTRY
from churnbench.eval.harness import RunHarness
from churnbench.eval.scoring import (
    TaskResult,
    classify_verdict,
    compute_t_eff,
    config_hash,
    task_result_from_dict,
    task_result_to_dict,
)
from churnbench.ledger.ledger import Ledger
from churnbench.tasks.schema import GoldAnswer, Task

# ── Shared fixtures ───────────────────────────────────────────────────────────

_T_PRIME = date(2024, 3, 10)
_T = date(2024, 3, 24)  # 14-day evaluation window


def _make_gold(value: Any, answer_type: str = "int") -> GoldAnswer:
    return GoldAnswer(value=value, answer_type=answer_type, resolver_ref="test_ref")


def _make_task(task_id: str, question: str = "How many?", answer_type: str = "int") -> Task:
    return Task(
        task_id=task_id,
        template_id="t1",
        intent="count",
        tier=1,
        question_text=question,
        params={"cc": "cc_001"},
        T=_T,
        answer_type=answer_type,
        resolver_ref="active_user_count_cc",
    )


def _noop_projector(ledger: Ledger, T: date, docs_dir: Path) -> None:
    """No-op injected projector for tests that don't need real DB projection."""


def _make_grounding_arm_with_stamped_registry(
    T_prime: date,
) -> GroundingArm:
    """Build a GroundingArm with all connections mocked and registry stamped at T_prime.

    This simulates the state after arm.setup() has run: all non-live entity classes
    have last_refresh = T_prime.  The refresh scheduler in the harness then drives
    the staleness lifecycle forward to T.
    """
    arm = GroundingArm()
    arm._lm = MagicMock()
    arm._pg_engine = MagicMock()
    arm._mongo_db = MagicMock()
    arm._saas = MagicMock()
    arm._embed = MagicMock()
    arm._docs_coll = MagicMock()
    arm._full_coll = None
    arm._registry = copy.deepcopy(ENTITY_REGISTRY)
    arm._staged_engine = create_engine("sqlite:///:memory:", future=True)
    create_staged_schema(arm._staged_engine)
    arm._t_prime = T_prime
    # Simulate setup() stamping: all non-live entities refreshed at T_prime
    for ec in arm._registry.values():
        if ec.tier != "live":
            ec.last_refresh = T_prime
    return arm


# ── Test 1: Lifecycle integration ─────────────────────────────────────────────


class TestLifecycle:
    """Critical test: verify hot/warm/cold last_refresh after the refresh scheduler runs.

    Window: T_prime = 2024-03-10, T = 2024-03-24 (14 days).

    Expected schedule for each tier:
      hot  (TTL=1):  first stale on day+2 → refreshed on Mar 12, 14, 16, 18, 20, 22, 24
                     final last_refresh = Mar 24 = T
      warm (TTL=7):  first stale on day+8 → refreshed on Mar 18 only
                     final last_refresh = Mar 18
      cold (TTL=30): never stale within 14-day window
                     final last_refresh = T_prime (unchanged)
    """

    def test_hot_entity_last_refresh_equals_T(self, tmp_path: Path) -> None:
        arm = _make_grounding_arm_with_stamped_registry(_T_PRIME)
        harness = RunHarness(
            run_dir=tmp_path,
            fabric_config=FabricConfig(),
            _projector=_noop_projector,
        )
        harness._schedule_refreshes(arm, Ledger(), _T_PRIME, _T)

        # assignments and user_status are both hot (TTL=1)
        assert arm._registry["assignments"].last_refresh == _T
        assert arm._registry["user_status"].last_refresh == _T

    def test_hot_entity_is_fresh_at_T(self, tmp_path: Path) -> None:
        arm = _make_grounding_arm_with_stamped_registry(_T_PRIME)
        harness = RunHarness(
            run_dir=tmp_path,
            fabric_config=FabricConfig(),
            _projector=_noop_projector,
        )
        harness._schedule_refreshes(arm, Ledger(), _T_PRIME, _T)
        assert not arm._registry["assignments"].is_stale_at(_T)

    def test_warm_entity_refreshed_at_day_18(self, tmp_path: Path) -> None:
        arm = _make_grounding_arm_with_stamped_registry(_T_PRIME)
        harness = RunHarness(
            run_dir=tmp_path,
            fabric_config=FabricConfig(),
            _projector=_noop_projector,
        )
        harness._schedule_refreshes(arm, Ledger(), _T_PRIME, _T)

        # TTL=7: first due on day +8 after T_prime (Mar 10 + 8 = Mar 18)
        expected_warm_refresh = date(2024, 3, 18)
        assert arm._registry["prices"].last_refresh == expected_warm_refresh
        assert arm._registry["cost_center_membership"].last_refresh == expected_warm_refresh
        assert not arm._registry["prices"].is_stale_at(_T)  # 6 days after refresh < TTL=7

    def test_cold_entity_never_refreshed_within_window(self, tmp_path: Path) -> None:
        arm = _make_grounding_arm_with_stamped_registry(_T_PRIME)
        harness = RunHarness(
            run_dir=tmp_path,
            fabric_config=FabricConfig(),
            _projector=_noop_projector,
        )
        harness._schedule_refreshes(arm, Ledger(), _T_PRIME, _T)

        # cold TTL=30: 14-day window never exceeds TTL → unchanged from setup
        assert arm._registry["contract_terms"].last_refresh == _T_PRIME
        assert arm._registry["vendor_dims"].last_refresh == _T_PRIME
        assert not arm._registry["vendor_dims"].is_stale_at(_T)

    def test_live_entities_have_no_last_refresh(self, tmp_path: Path) -> None:
        arm = _make_grounding_arm_with_stamped_registry(_T_PRIME)
        harness = RunHarness(
            run_dir=tmp_path,
            fabric_config=FabricConfig(),
            _projector=_noop_projector,
        )
        harness._schedule_refreshes(arm, Ledger(), _T_PRIME, _T)

        # Live-tier entities are never staged → last_refresh always None
        for name in ("consumption_facts", "utilization_current", "tickets"):
            assert arm._registry[name].last_refresh is None

    def test_non_grounding_arm_schedule_is_noop(self, tmp_path: Path) -> None:
        """schedule_refreshes must be a no-op for all non-grounding arms."""
        from churnbench.arms.naive import NaiveArm

        arm = NaiveArm()
        harness = RunHarness(
            run_dir=tmp_path,
            fabric_config=FabricConfig(),
            _projector=_noop_projector,
        )
        # Should complete without error or exception
        harness._schedule_refreshes(arm, Ledger(), _T_PRIME, _T)


# ── Test 2: T_eff derivation ──────────────────────────────────────────────────


class TestTeff:
    def test_naive_always_returns_T(self) -> None:
        assert compute_t_eff("naive", [], _T_PRIME, _T) == _T

    def test_classic_rag_always_returns_T_prime(self) -> None:
        assert compute_t_eff("classic_rag", [], _T_PRIME, _T) == _T_PRIME

    def test_hierarchical_docs_worker_gives_T_prime(self) -> None:
        trace = [
            {"role": "docs_worker", "source_ts": _T_PRIME.isoformat()},
            {"role": "sql_worker", "source_ts": "live"},
        ]
        assert compute_t_eff("hierarchical", trace, _T_PRIME, _T) == _T_PRIME

    def test_hierarchical_all_live_gives_T(self) -> None:
        trace = [
            {"role": "sql_worker", "source_ts": "live"},
            {"role": "mongo_worker", "source_ts": "live"},
        ]
        assert compute_t_eff("hierarchical", trace, _T_PRIME, _T) == _T

    def test_hierarchical_no_workers_fallback_to_T_prime(self) -> None:
        trace = [{"role": "direct_answer", "content": "42"}]
        assert compute_t_eff("hierarchical", trace, _T_PRIME, _T) == _T_PRIME

    def test_hierarchical_min_of_mixed_sources(self) -> None:
        intermediate = date(2024, 3, 15)
        trace = [
            {"role": "docs_worker", "source_ts": intermediate.isoformat()},
            {"role": "sql_worker", "source_ts": "live"},
        ]
        assert compute_t_eff("hierarchical", trace, _T_PRIME, _T) == intermediate

    def test_grounding_staged_gives_min_last_refresh(self) -> None:
        earlier = date(2024, 3, 14)
        later = date(2024, 3, 20)
        trace = [
            {"role": "retrieval", "staged_vs_live": "staged", "last_refresh": later.isoformat()},
            {"role": "retrieval", "staged_vs_live": "staged", "last_refresh": earlier.isoformat()},
        ]
        assert compute_t_eff("grounding", trace, _T_PRIME, _T) == earlier

    def test_grounding_live_route_contributes_T(self) -> None:
        trace = [
            {"role": "retrieval", "staged_vs_live": "live", "last_refresh": None},
        ]
        assert compute_t_eff("grounding", trace, _T_PRIME, _T) == _T

    def test_grounding_mixed_staged_and_live(self) -> None:
        staged_refresh = date(2024, 3, 16)
        trace = [
            {"role": "retrieval", "staged_vs_live": "staged",
             "last_refresh": staged_refresh.isoformat()},
            {"role": "retrieval", "staged_vs_live": "live", "last_refresh": None},
        ]
        # min(staged_refresh, T) = staged_refresh
        assert compute_t_eff("grounding", trace, _T_PRIME, _T) == staged_refresh

    def test_grounding_no_retrieval_entries_fallback(self) -> None:
        trace = [{"role": "need_resolution", "entity_classes": ["user_status"]}]
        assert compute_t_eff("grounding", trace, _T_PRIME, _T) == _T_PRIME

    def test_grounding_ablation_name_resolves_correctly(self) -> None:
        # Ablation name still starts with "grounding" → same T_eff logic
        trace = [
            {"role": "retrieval", "staged_vs_live": "live", "last_refresh": None},
        ]
        assert compute_t_eff("grounding_no_freshness_tiers", trace, _T_PRIME, _T) == _T


# ── Test 3: Verdict classification ───────────────────────────────────────────


class TestVerdicts:
    def test_correct_int(self) -> None:
        gold_T = _make_gold(42)
        gold_Teff = _make_gold(40)
        assert classify_verdict("42", "int", gold_T, gold_Teff, _T_PRIME, _T) == "correct"

    def test_freshness_error_int(self) -> None:
        # Answer matches T_prime gold but not T gold
        gold_T = _make_gold(42)
        gold_Teff = _make_gold(40)
        assert classify_verdict("40", "int", gold_T, gold_Teff, _T_PRIME, _T) == "freshness_error"

    def test_reasoning_error_wrong_at_both(self) -> None:
        gold_T = _make_gold(42)
        gold_Teff = _make_gold(40)
        assert classify_verdict("99", "int", gold_T, gold_Teff, _T_PRIME, _T) == "reasoning_error"

    def test_no_freshness_error_when_t_eff_equals_T(self) -> None:
        """When T_eff = T (naive arm), wrong at T cannot be a freshness error."""
        gold_T = _make_gold(42)
        gold_Teff = _make_gold(42)  # same as T since T_eff = T
        assert classify_verdict("40", "int", gold_T, gold_Teff, _T, _T) == "reasoning_error"

    def test_parse_failure_non_numeric_int(self) -> None:
        gold_T = _make_gold(42)
        gold_Teff = _make_gold(42)
        assert classify_verdict("none found", "int", gold_T, gold_Teff, _T_PRIME, _T) == "parse_failure"

    def test_correct_float_within_1pct_tolerance(self) -> None:
        gold_T = _make_gold(100.0, "float")
        gold_Teff = _make_gold(90.0, "float")
        # 100.5 is within 1% of 100.0
        assert classify_verdict("$100.50", "float", gold_T, gold_Teff, _T, _T) == "correct"

    def test_freshness_error_float_stale_answer(self) -> None:
        gold_T = _make_gold(100.0, "float")
        gold_Teff = _make_gold(89.0, "float")
        # 89.0 is wrong at T but right at T_prime (gold_Teff = 89.0)
        assert classify_verdict("89.0", "float", gold_T, gold_Teff, _T_PRIME, _T) == "freshness_error"

    def test_correct_list_exact(self) -> None:
        gold_T = _make_gold(["a", "b", "c"], "list[str]")
        gold_Teff = _make_gold(["a", "b"], "list[str]")
        assert classify_verdict('["a", "b", "c"]', "list[str]", gold_T, gold_Teff, _T, _T) == "correct"

    def test_correct_str_case_insensitive(self) -> None:
        gold_T = _make_gold("active", "str")
        gold_Teff = _make_gold("expired", "str")
        assert classify_verdict("Active", "str", gold_T, gold_Teff, _T, _T) == "correct"


# ── Test 4: Naive arm structural validity ─────────────────────────────────────


class TestNaiveValidity:
    """Freshness errors are structurally impossible for the naive arm.

    For naive, T_eff = T. So:
      gold_at_T_eff ≡ gold_at_T
      wrong at T → also wrong at T_eff (same gold, same answer)
      → always reasoning_error, never freshness_error.
    """

    def test_wrong_answer_is_reasoning_error_not_freshness_error(self) -> None:
        gold_at_T = _make_gold(42)
        # For naive, gold_at_T_eff == gold_at_T since T_eff = T
        gold_at_t_eff = _make_gold(42)
        assert classify_verdict("40", "int", gold_at_T, gold_at_t_eff, _T, _T) == "reasoning_error"

    def test_batch_of_wrong_answers_have_zero_freshness_errors(self) -> None:
        from churnbench.eval.scoring import compute_summary

        results: list[TaskResult] = []
        for i in range(10):
            gold_T = _make_gold(i + 1)
            gold_Teff = _make_gold(i + 1)  # same: T_eff = T for naive
            verdict = classify_verdict(
                str(i + 100),  # all wrong
                "int", gold_T, gold_Teff, _T, _T  # T_eff = T
            )
            results.append(
                TaskResult(
                    task_id=f"t{i}",
                    arm="naive",
                    task_tier=1,
                    task_intent="count",
                    question_text="?",
                    answer_raw=str(i + 100),
                    answer_parsed=i + 100,
                    gold=i + 1,
                    gold_at_t_eff=i + 1,
                    verdict=verdict,
                    t_eff=_T.isoformat(),
                    cost_usd=0.001,
                    latency_s=0.1,
                    input_tokens=50,
                    output_tokens=10,
                    embedding_tokens=0,
                    attribution={},
                    tool_calls=[],
                )
            )
        m = compute_summary(results)
        assert m.n_freshness_error == 0
        assert m.freshness_error_rate == 0.0


# ── Test 5: Checkpoint / resume ───────────────────────────────────────────────


class TestCheckpointResume:
    def _make_task_result(self, task_id: str, verdict: str = "correct") -> TaskResult:
        return TaskResult(
            task_id=task_id,
            arm="naive",
            task_tier=1,
            task_intent="count",
            question_text="Test?",
            answer_raw="42",
            answer_parsed=42,
            gold=42,
            gold_at_t_eff=42,
            verdict=verdict,
            t_eff=_T.isoformat(),
            cost_usd=0.001,
            latency_s=0.1,
            input_tokens=50,
            output_tokens=10,
            embedding_tokens=0,
            attribution={},
            tool_calls=[],
        )

    def test_load_checkpoint_returns_correct_ids_and_results(self, tmp_path: Path) -> None:
        harness = RunHarness(
            run_dir=tmp_path,
            fabric_config=FabricConfig(),
            _projector=_noop_projector,
        )
        ckpt_path = tmp_path / "test.ckpt.json"
        results = [self._make_task_result("task_001"), self._make_task_result("task_002", "freshness_error")]
        harness._write_checkpoint(ckpt_path, results)

        loaded_ids, loaded_results = harness._load_checkpoint(ckpt_path)
        assert loaded_ids == {"task_001", "task_002"}
        assert len(loaded_results) == 2
        assert loaded_results[0].task_id == "task_001"
        assert loaded_results[0].verdict == "correct"
        assert loaded_results[1].task_id == "task_002"
        assert loaded_results[1].verdict == "freshness_error"

    def test_load_missing_checkpoint_returns_empty(self, tmp_path: Path) -> None:
        harness = RunHarness(
            run_dir=tmp_path,
            fabric_config=FabricConfig(),
            _projector=_noop_projector,
        )
        ids, results = harness._load_checkpoint(tmp_path / "nonexistent.ckpt.json")
        assert ids == set()
        assert results == []

    def test_task_result_roundtrip_through_json(self) -> None:
        original = self._make_task_result("task_xyz", "freshness_error")
        d = task_result_to_dict(original)
        restored = task_result_from_dict(d)
        assert restored.task_id == original.task_id
        assert restored.verdict == original.verdict
        assert restored.gold == original.gold
        assert restored.t_eff == original.t_eff

    def test_checkpoint_is_deleted_after_successful_write(self, tmp_path: Path) -> None:
        """The checkpoint file must be removed when results are written successfully."""
        harness = RunHarness(
            run_dir=tmp_path,
            fabric_config=FabricConfig(),
            _projector=_noop_projector,
        )
        ckpt_path = tmp_path / "arm__abc123.ckpt.json"
        result_path = tmp_path / "arm__abc123.json"
        results = [self._make_task_result("t1")]
        from churnbench.eval.scoring import compute_summary

        harness._write_checkpoint(ckpt_path, results)
        assert ckpt_path.exists()

        harness._write_results(result_path, "arm", "abc123", results, compute_summary(results))
        ckpt_path.unlink(missing_ok=True)  # harness.run() does this
        assert not ckpt_path.exists()
        assert result_path.exists()


# ── Test 6: Config hash stability ────────────────────────────────────────────


class TestConfigHash:
    def _tasks(self, *ids: str) -> list[Task]:
        return [_make_task(tid) for tid in ids]

    def test_same_args_yield_same_hash(self) -> None:
        tasks = self._tasks("t1", "t2", "t3")
        h1 = config_hash("grounding", {}, "claude-sonnet-4-6", _T_PRIME, _T, 42, tasks)
        h2 = config_hash("grounding", {}, "claude-sonnet-4-6", _T_PRIME, _T, 42, tasks)
        assert h1 == h2

    def test_different_arm_name_yields_different_hash(self) -> None:
        tasks = self._tasks("t1")
        h1 = config_hash("naive", {}, "claude-sonnet-4-6", _T_PRIME, _T, 42, tasks)
        h2 = config_hash("grounding", {}, "claude-sonnet-4-6", _T_PRIME, _T, 42, tasks)
        assert h1 != h2

    def test_different_T_yields_different_hash(self) -> None:
        tasks = self._tasks("t1")
        h1 = config_hash("naive", {}, "claude-sonnet-4-6", _T_PRIME, _T, 42, tasks)
        h2 = config_hash("naive", {}, "claude-sonnet-4-6", _T_PRIME, _T + timedelta(days=5), 42, tasks)
        assert h1 != h2

    def test_different_T_prime_yields_different_hash(self) -> None:
        tasks = self._tasks("t1")
        h1 = config_hash("naive", {}, "claude-sonnet-4-6", _T_PRIME, _T, 42, tasks)
        h2 = config_hash("naive", {}, "claude-sonnet-4-6", _T_PRIME + timedelta(days=1), _T, 42, tasks)
        assert h1 != h2

    def test_different_seed_yields_different_hash(self) -> None:
        tasks = self._tasks("t1")
        h1 = config_hash("naive", {}, "claude-sonnet-4-6", _T_PRIME, _T, 42, tasks)
        h2 = config_hash("naive", {}, "claude-sonnet-4-6", _T_PRIME, _T, 99, tasks)
        assert h1 != h2

    def test_different_task_set_yields_different_hash(self) -> None:
        h1 = config_hash("naive", {}, "claude-sonnet-4-6", _T_PRIME, _T, 42, self._tasks("t1"))
        h2 = config_hash("naive", {}, "claude-sonnet-4-6", _T_PRIME, _T, 42, self._tasks("t1", "t2"))
        assert h1 != h2

    def test_flags_affect_hash(self) -> None:
        tasks = self._tasks("t1")
        h1 = config_hash("grounding", {}, "claude-sonnet-4-6", _T_PRIME, _T, 42, tasks)
        h2 = config_hash("grounding", {"no_freshness_tiers": True}, "claude-sonnet-4-6", _T_PRIME, _T, 42, tasks)
        assert h1 != h2

    def test_hash_is_16_hex_chars(self) -> None:
        h = config_hash("naive", {}, "claude-sonnet-4-6", _T_PRIME, _T, 42, self._tasks("t1"))
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)
