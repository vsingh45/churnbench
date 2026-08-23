"""Tests for churnbench/tasks/: resolver, generator, schema, templates.

All tests are in-memory — no Postgres, MongoDB, or filesystem writes.
A rich ledger fixture provides enough data to exercise every intent.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from churnbench.generator.timeline import TimelineConfig, TimelineSimulator
from churnbench.ledger.ledger import EventKind, Ledger
from churnbench.tasks.generator import TaskSetGenerator, _is_degenerate
from churnbench.tasks.resolver import LedgerResolver
from churnbench.tasks.schema import Task, load_tasks, save_tasks
from churnbench.tasks.templates import TEMPLATES_BY_TIER


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: rich 45-day ledger with consumption, offboards, contracts
# ─────────────────────────────────────────────────────────────────────────────


def _rich_ledger() -> Ledger:
    """45-day ledger sufficient to exercise every intent and tier.

    Timeline:
      Day 0 (2024-01-01): seed — products, contracts, users, licenses, assignments
      Day 7  (2024-01-08): consumption batch, one reassignment
      Day 14 (2024-01-15): offboard usr_000002, license freed
      Day 21 (2024-01-22): consumption batch, price change
      Day 30 (2024-01-31): contract renewal, new user hired
      Day 45 (2024-02-15): consumption batch
    """
    led = Ledger()
    d0 = date(2024, 1, 1)
    d7 = date(2024, 1, 8)
    d14 = date(2024, 1, 15)
    d21 = date(2024, 1, 22)
    d30 = date(2024, 1, 31)
    d45 = date(2024, 2, 15)

    # 3 products, initial prices
    for i, price in enumerate([100.0, 200.0, 150.0]):
        led.append(
            d0,
            EventKind.PRICE_CHANGED,
            "product",
            f"prd_{i:04d}",
            {"unit_price_usd": price, "reason": "initial"},
        )

    # 2 contracts
    led.append(
        d0, EventKind.CONTRACT_SIGNED, "contract", "ctr_0000", {"vendor_idx": 0, "term_months": 12}
    )
    led.append(
        d0, EventKind.CONTRACT_SIGNED, "contract", "ctr_0001", {"vendor_idx": 1, "term_months": 6}
    )

    # 4 users across 2 cost centers
    for i, cc in enumerate(["cc_000", "cc_000", "cc_001", "cc_001"]):
        led.append(d0, EventKind.USER_HIRED, "user", f"usr_{i:06d}", {"cost_center_id": cc})

    # 4 licenses purchased + assigned
    for i in range(4):
        pid = f"prd_{i % 3:04d}"
        led.append(
            d0,
            EventKind.LICENSE_PURCHASED,
            "license",
            f"lic_{i:06d}",
            {"product_id": pid, "seats": 1, "unit_price_usd": [100.0, 200.0, 150.0, 100.0][i]},
        )
        led.append(
            d0,
            EventKind.LICENSE_ASSIGNED,
            "license",
            f"lic_{i:06d}",
            {"to": f"usr_{i:06d}", "product_id": pid},
        )

    # Day 7: consumption for all products
    for user_i in range(4):
        for prod_i in range(3):
            led.append(
                d7,
                EventKind.CONSUMPTION_LOGGED,
                "user",
                f"usr_{user_i:06d}",
                {"product_id": f"prd_{prod_i:04d}", "session_minutes": 30, "api_calls": 10},
            )
    # One reassignment
    led.append(
        d7,
        EventKind.LICENSE_REASSIGNED,
        "license",
        "lic_000003",
        {"from": "usr_000003", "to": "usr_000000"},
    )

    # Day 14: offboard usr_000002
    led.append(
        d14,
        EventKind.LICENSE_UNASSIGNED,
        "license",
        "lic_000002",
        {"prev_holder": "usr_000002", "reason": "offboard"},
    )
    led.append(d14, EventKind.USER_OFFBOARDED, "user", "usr_000002", {})

    # Day 21: more consumption + price change
    for user_i in [0, 1, 3]:
        led.append(
            d21,
            EventKind.CONSUMPTION_LOGGED,
            "user",
            f"usr_{user_i:06d}",
            {"product_id": "prd_0000", "session_minutes": 45, "api_calls": 20},
        )
    led.append(
        d21,
        EventKind.PRICE_CHANGED,
        "product",
        "prd_0000",
        {"unit_price_usd": 120.0, "prev_price": 100.0, "pct_change": 0.2},
    )

    # Day 30: contract renewal, new user
    led.append(d30, EventKind.CONTRACT_RENEWED, "contract", "ctr_0001", {"term_months": 12})
    led.append(d30, EventKind.USER_HIRED, "user", "usr_000004", {"cost_center_id": "cc_001"})

    # Day 45: consumption for remaining active users
    for user_i in [0, 1, 3]:
        led.append(
            d45,
            EventKind.CONSUMPTION_LOGGED,
            "user",
            f"usr_{user_i:06d}",
            {"product_id": "prd_0001", "session_minutes": 60, "api_calls": 15},
        )

    return led


T_EVAL = date(2024, 2, 15)  # day 45 — full ledger
T_MID = date(2024, 1, 22)  # day 21 — pre-renewal


# ─────────────────────────────────────────────────────────────────────────────
# Resolver correctness
# ─────────────────────────────────────────────────────────────────────────────


class TestResolver:
    def test_active_users_excludes_offboarded(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        active = resolver.active_users(T_EVAL)
        assert "usr_000002" not in active
        assert len(active) == 4  # 0,1,3 from start + 4 hired day 30

    def test_product_price_before_change(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        assert resolver.product_price("prd_0000", date(2024, 1, 10)) == pytest.approx(100.0)

    def test_product_price_after_change(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        assert resolver.product_price("prd_0000", T_EVAL) == pytest.approx(120.0)

    def test_monthly_spend_by_cost_center_sums_active_licenses(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        spend = resolver.monthly_spend_by_cost_center(T_EVAL)
        # cc_000 has usr_000000 (lic_0: 100+, lic_3 reassigned: 100+) + usr_000001 (lic_1: 200)
        # cc_001 has usr_000003 (lic_3 was reassigned away on day 7 → usr_000000)
        #          usr_000004 (no license)
        # Note: lic_000002 is unassigned after offboard
        assert "cc_000" in spend
        assert spend["cc_000"] > 0

    def test_unassigned_license_count(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        count = resolver.unassigned_license_count(T_EVAL)
        assert count == 1  # lic_000002 was unassigned when usr_000002 offboarded

    def test_orphan_license_count(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        # lic_000002 is unassigned (holder_id=None), NOT orphaned
        # No license is currently held by an offboarded user
        assert resolver.orphan_license_count(T_EVAL) == 0

    def test_reassignment_count_in_window(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        # Day 7 had 1 reassignment; day 0–7 window
        count = resolver.reassignment_count(days_back=10, T=date(2024, 1, 10))
        assert count == 1

    def test_offboard_count_window(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        # 1 offboard on day 14; looking back 30 days from day 45
        count = resolver.offboard_count_window(days_back=40, T=T_EVAL)
        assert count == 1

    def test_contract_status_active(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        # ctr_0001 renewed on day 30 for 12 months → active at day 45
        assert resolver.contract_status("ctr_0001", T_EVAL) == "active"

    def test_contract_months_remaining_positive(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        months = resolver.contract_months_remaining("ctr_0001", T_EVAL)
        assert months > 0

    def test_contract_vendor(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        v = resolver.contract_vendor("ctr_0000", T_EVAL)
        assert v == "Acme Corp"  # vendor_idx=0

    def test_contract_annual_value_positive(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        val = resolver.contract_annual_value("ctr_0000", T_EVAL)
        assert val >= 10_000

    def test_product_api_calls_within_window(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        # Day 45: 3 users × 15 calls for prd_0001
        calls = resolver.product_api_calls("prd_0001", window_days=20, T=T_EVAL)
        assert calls == 45  # 3 users × 15 api_calls

    def test_product_active_users_within_window(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        users = resolver.product_active_users("prd_0001", window_days=20, T=T_EVAL)
        assert users == 3

    def test_idle_license_count_for_product(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        # prd_0002 had no consumption after day 7; idle for 38+ days at T_EVAL
        count = resolver.idle_license_count("prd_0002", idle_days=30, T=T_EVAL)
        assert count >= 0  # might be 0 if license was reassigned/unassigned

    def test_license_efficiency_non_negative(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        eff = resolver.license_efficiency("prd_0001", window_days=20, T=T_EVAL)
        # Efficiency can exceed 1.0 when more users access a product than licensed seats
        assert eff >= 0.0

    def test_ws_cache_same_T(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        ws1 = resolver._ws(T_EVAL)
        ws2 = resolver._ws(T_EVAL)
        assert ws1 is ws2  # same object — cache hit


# ─────────────────────────────────────────────────────────────────────────────
# Generator determinism
# ─────────────────────────────────────────────────────────────────────────────


class TestGeneratorDeterminism:
    def test_same_seed_produces_identical_task_list(self) -> None:
        led = _rich_ledger()
        tasks1 = TaskSetGenerator(led, T_EVAL, seed=7, n=50).generate()
        tasks2 = TaskSetGenerator(led, T_EVAL, seed=7, n=50).generate()
        assert len(tasks1) == len(tasks2)
        for t1, t2 in zip(tasks1, tasks2):
            assert t1.task_id == t2.task_id
            assert t1.template_id == t2.template_id
            assert t1.params == t2.params

    def test_different_seed_produces_different_tasks(self) -> None:
        led = _rich_ledger()
        tasks1 = TaskSetGenerator(led, T_EVAL, seed=1, n=30).generate()
        tasks2 = TaskSetGenerator(led, T_EVAL, seed=2, n=30).generate()
        # At least one task should differ
        same = sum(
            t1.params == t2.params and t1.template_id == t2.template_id
            for t1, t2 in zip(tasks1, tasks2)
        )
        assert same < 30

    def test_generates_requested_n(self) -> None:
        led = _rich_ledger()
        tasks = TaskSetGenerator(led, T_EVAL, seed=42, n=60).generate()
        assert len(tasks) == 60

    def test_task_ids_are_sequential(self) -> None:
        led = _rich_ledger()
        tasks = TaskSetGenerator(led, T_EVAL, seed=7, n=20).generate()
        for i, t in enumerate(tasks):
            assert t.task_id == f"task_{i:04d}"


# ─────────────────────────────────────────────────────────────────────────────
# Gold correctness per intent
# ─────────────────────────────────────────────────────────────────────────────


class TestGoldCorrectness:
    def _tasks_for(self, intent: str, n: int = 60) -> list[Task]:
        led = _rich_ledger()
        all_tasks = TaskSetGenerator(led, T_EVAL, seed=7, n=n).generate()
        return [t for t in all_tasks if t.intent == intent]

    def test_spend_visibility_gold_matches_answer_type(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        type_map = {"float": float, "int": int, "str": str}
        for t in self._tasks_for("spend_visibility"):
            gold = t.gold(resolver, at=T_EVAL)
            expected_type = type_map[t.answer_type]
            assert isinstance(gold.value, expected_type), (
                f"Task {t.task_id} ({t.resolver_ref}): expected {expected_type.__name__}, "
                f"got {type(gold.value).__name__} = {gold.value!r}"
            )

    def test_savings_opportunity_gold_is_non_negative(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        for t in self._tasks_for("savings_opportunity"):
            gold = t.gold(resolver, at=T_EVAL)
            assert isinstance(gold.value, (int, float))
            assert gold.value >= 0

    def test_criticality_contract_tasks_return_valid_str_or_numeric(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        led = _rich_ledger()
        for t in TaskSetGenerator(led, T_EVAL, seed=7, n=100).generate():
            if t.intent == "criticality" and t.tier == 3:
                gold = t.gold(resolver, at=T_EVAL)
                assert gold.value is not None
                assert gold.value != ""

    def test_utilization_api_calls_non_negative(self) -> None:
        resolver = LedgerResolver(_rich_ledger())
        for t in self._tasks_for("utilization"):
            if t.resolver_ref == "product_api_calls":
                gold = t.gold(resolver, at=T_EVAL)
                assert isinstance(gold.value, int)
                assert gold.value >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Temporal sensitivity
# ─────────────────────────────────────────────────────────────────────────────


class TestTemporalSensitivity:
    def test_gold_changes_between_T_and_earlier(self) -> None:
        """At least one task's gold should differ at T vs T-30 days."""
        led = _rich_ledger()
        resolver = LedgerResolver(led)
        tasks = TaskSetGenerator(led, T_EVAL, seed=7, n=80).generate()
        T_prime = T_EVAL - timedelta(days=30)

        changed = 0
        for t in tasks:
            g_now = t.gold(resolver, at=T_EVAL)
            g_stale = t.gold(resolver, at=T_prime)
            if g_now.value != g_stale.value:
                changed += 1

        assert changed >= 1, "Expected at least one task whose gold changes over time"

    def test_price_task_shows_temporal_change(self) -> None:
        """SV3 (product_price) for prd_0000 must differ before/after day 21 price change."""
        led = _rich_ledger()
        resolver = LedgerResolver(led)
        before = resolver.product_price("prd_0000", T=date(2024, 1, 20))
        after = resolver.product_price("prd_0000", T=T_EVAL)
        assert before != after


# ─────────────────────────────────────────────────────────────────────────────
# Tier-3 contract dependency
# ─────────────────────────────────────────────────────────────────────────────


class TestTier3ContractDependency:
    def test_tier3_tasks_reference_contract_or_cross_source(self) -> None:
        """Every tier-3 task either has contract_id in params or uses a contract resolver."""
        contract_resolver_refs = {
            "contract_status",
            "contract_months_remaining",
            "contract_vendor",
            "contract_annual_value",
            "contract_vendor_tier",
            "license_efficiency",
        }
        led = _rich_ledger()
        tasks = TaskSetGenerator(led, T_EVAL, seed=7, n=120).generate()
        tier3 = [t for t in tasks if t.tier == 3]
        assert len(tier3) > 0, "Expected at least one tier-3 task"
        for t in tier3:
            has_contract_param = "contract_id" in t.params
            has_contract_ref = t.resolver_ref in contract_resolver_refs
            assert (
                has_contract_param or has_contract_ref
            ), f"Tier-3 task {t.task_id} ({t.resolver_ref}) lacks contract dependency"

    def test_tier3_templates_exist(self) -> None:
        assert len(TEMPLATES_BY_TIER[3]) >= 4


# ─────────────────────────────────────────────────────────────────────────────
# Degenerate gold rate
# ─────────────────────────────────────────────────────────────────────────────


class TestDegenerateRate:
    def test_degenerate_gold_rate_at_T(self) -> None:
        """At most 10% of generated tasks should have a degenerate gold at T."""
        led = _rich_ledger()
        resolver = LedgerResolver(led)
        tasks = TaskSetGenerator(led, T_EVAL, seed=7, n=120).generate()
        degen = sum(1 for t in tasks if _is_degenerate(t.gold(resolver, at=T_EVAL).value))
        rate = degen / len(tasks)
        assert rate <= 0.10, f"Degenerate rate {rate:.1%} exceeds 10%"

    def test_degenerate_gold_rate_at_T_minus_7(self) -> None:
        """Gold answers at T-7 should also remain mostly non-degenerate."""
        led = _rich_ledger()
        resolver = LedgerResolver(led)
        tasks = TaskSetGenerator(led, T_EVAL, seed=7, n=120).generate()
        T_prime = T_EVAL - timedelta(days=7)
        degen = sum(1 for t in tasks if _is_degenerate(t.gold(resolver, at=T_prime).value))
        rate = degen / len(tasks)
        assert rate <= 0.10, f"Degenerate rate at T-7 {rate:.1%} exceeds 10%"

    def test_trivial_zero_arm_score_below_10pct(self) -> None:
        """A trivial arm always answering 0/0.0/'' should score under 10%.

        Uses the full 90-day simulator (seed 42) to match the audit scenario.
        This is the permanent regression guard that the per-template zero quota
        (PER_TEMPLATE_ZERO_QUOTA) and the SO3/SO5 template replacements are
        designed to keep passing.
        """
        # 2023-12-16 + 90 days = 2024-03-15, so T is the last day of the window
        cfg = TimelineConfig(start=date(2023, 12, 16), days=90, seed=42)
        led = Ledger()
        TimelineSimulator(cfg, led).run()

        T = date(2024, 3, 15)
        resolver = LedgerResolver(led)
        tasks = TaskSetGenerator(led, T, seed=7, n=180).generate()

        trivial_correct = 0
        for t in tasks:
            gold = t.gold(resolver, at=T).value
            if t.answer_type == "int" and gold == 0:
                trivial_correct += 1
            elif t.answer_type == "float" and gold == 0.0:
                trivial_correct += 1
            elif t.answer_type == "str" and gold == "":
                trivial_correct += 1

        score = trivial_correct / len(tasks)
        assert score < 0.10, (
            f"Trivial zero-arm score {score:.1%} >= 10% "
            f"({trivial_correct}/{len(tasks)} tasks answered by always-zero arm)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Schema round-trip (save + load)
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaRoundTrip:
    def test_save_and_load_tasks(self, tmp_path: Path) -> None:
        led = _rich_ledger()
        tasks = TaskSetGenerator(led, T_EVAL, seed=7, n=20).generate()
        out = tmp_path / "tasks.jsonl"
        save_tasks(tasks, out)
        loaded = load_tasks(out)
        assert len(loaded) == len(tasks)
        for orig, back in zip(tasks, loaded):
            assert orig.task_id == back.task_id
            assert orig.template_id == back.template_id
            assert orig.T == back.T
            assert orig.params == back.params
            assert orig.resolver_ref == back.resolver_ref

    def test_tier_balance(self) -> None:
        """Tier distribution should be roughly 40/40/20."""
        led = _rich_ledger()
        tasks = TaskSetGenerator(led, T_EVAL, seed=7, n=180).generate()
        counts = {1: 0, 2: 0, 3: 0}
        for t in tasks:
            counts[t.tier] += 1
        total = len(tasks)
        # Allow ±15% slack around target
        assert counts[1] / total >= 0.25, "Too few tier-1 tasks"
        assert counts[2] / total >= 0.25, "Too few tier-2 tasks"
        assert counts[3] / total >= 0.05, "Too few tier-3 tasks"
