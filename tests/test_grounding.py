"""Tests for churnbench/arms/grounding/ — Arm 4 (grounding architecture).

All tests are offline: no real Postgres/MongoDB/SaaS connections.
LLM calls use _FakeLM; embeddings use _FakeEmbedModel; SQLite is in-memory.
"""

from __future__ import annotations

import copy
import json
from datetime import date, timedelta
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import create_engine, text

from churnbench.arms.base import cost_usd
from churnbench.arms.grounding.arm import GroundingArm, _parse_need_json
from churnbench.arms.grounding.etl import create_staged_schema, refresh_due, setup
from churnbench.arms.grounding.router import (
    STAGED_SQL_TEMPLATES,
    WAREHOUSE_SQL_TEMPLATES,
    decide,
    keyword_entity_classes,
)
from churnbench.arms.grounding.semantic_model import (
    ENTITY_REGISTRY,
    MEASURE_TO_ENTITY,
    EntityClass,
)
from churnbench.tasks.schema import Task

# ── Shared helpers ────────────────────────────────────────────────────────────

_T_PRIME = date(2024, 1, 1)
_T_EVAL = date(2024, 1, 15)


def _make_task(
    answer_type: str = "int",
    question: str = "How many active users are in cc_000?",
) -> Task:
    return Task(
        task_id="task_g0001",
        template_id="SV2",
        intent="spend_visibility",
        tier=1,
        question_text=question,
        params={"cc": "cc_000"},
        T=_T_EVAL,
        answer_type=answer_type,
        resolver_ref="active_user_count_cc",
    )


def _ai_message(content: str, input_tokens: int = 50, output_tokens: int = 10) -> Any:
    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


class _FakeEmbedModel:
    _VOCAB = [
        "license",
        "cost",
        "user",
        "product",
        "contract",
        "spend",
        "vendor",
        "active",
        "assigned",
        "monthly",
        "ticket",
        "utilization",
    ]

    def encode(self, texts: list[str], **_: Any) -> Any:
        rows = []
        for txt in texts:
            words = set(txt.lower().split())
            vec = [1.0 if kw in words else 0.0 for kw in self._VOCAB]
            norm = max(sum(v**2 for v in vec) ** 0.5, 1e-8)
            rows.append([v / norm for v in vec])
        return np.array(rows, dtype=np.float32)


class _FakeLM:
    def __init__(self, responses: list[Any]) -> None:
        self._resp = iter(responses)
        self.model_name = "claude-sonnet-4-6"

    def invoke(self, _messages: Any, **_kw: Any) -> Any:
        return next(self._resp)


def _make_staged_engine(users: list[dict] | None = None) -> Any:
    """Return an in-memory SQLite engine with the staged schema populated."""
    engine = create_engine("sqlite:///:memory:", future=True)
    create_staged_schema(engine)
    if users:
        with engine.connect() as conn:
            for u in users:
                conn.execute(
                    text(
                        "INSERT INTO staged_users "
                        "(user_id, cost_center_id, active, hired_at, staged_at) "
                        "VALUES (:uid, :cc, :active, :hired, :ts)"
                    ),
                    {
                        "uid": u.get("user_id", "u1"),
                        "cc": u.get("cost_center_id", "cc_000"),
                        "active": 1 if u.get("active", True) else 0,
                        "hired": str(u.get("hired_at", "2020-01-01")),
                        "ts": _T_PRIME.isoformat(),
                    },
                )
            conn.commit()
    return engine


def _minimal_arm(lm: Any = None, staged_engine: Any = None) -> GroundingArm:
    """Build a GroundingArm with all connections mocked — no setup() call needed."""
    engine = staged_engine or _make_staged_engine()
    real_lm = lm or MagicMock()
    arm = GroundingArm(_staged_engine=engine)
    # Bypass setup() by setting private attributes directly
    arm._lm = real_lm
    arm._model = "claude-sonnet-4-6"
    arm._pg_engine = MagicMock()
    arm._mongo_db = MagicMock()
    arm._saas = MagicMock()
    arm._embed = _FakeEmbedModel()
    arm._staged_engine = engine
    arm._docs_coll = MagicMock()
    arm._full_coll = None
    arm._registry = copy.deepcopy(ENTITY_REGISTRY)
    # Stamp T_EVAL so entities appear fresh during evaluation
    # (T_PRIME is 14 days before T_EVAL — stale for hot TTL=1d; tests that need
    #  stale behaviour set last_refresh explicitly after calling _minimal_arm)
    for ec in arm._registry.values():
        if ec.tier != "live":
            ec.last_refresh = _T_EVAL
    arm._embedding_tokens = 0
    arm._t_prime = _T_PRIME
    return arm


# ═════════════════════════════════════════════════════════════════════════════
# 1. Semantic model registry
# ═════════════════════════════════════════════════════════════════════════════


class TestSemanticModelRegistry:
    def test_all_entities_have_origin(self) -> None:
        valid = {"postgres", "mongo", "saas", "docs"}
        for name, ec in ENTITY_REGISTRY.items():
            assert ec.origin in valid, f"{name}.origin invalid: {ec.origin}"

    def test_all_entities_have_tier(self) -> None:
        valid = {"hot", "warm", "cold", "live"}
        for name, ec in ENTITY_REGISTRY.items():
            assert ec.tier in valid, f"{name}.tier invalid: {ec.tier}"

    def test_hot_entities_ttl_is_1_day(self) -> None:
        hot = [ec for ec in ENTITY_REGISTRY.values() if ec.tier == "hot"]
        assert hot, "no hot entities"
        for ec in hot:
            assert ec.ttl_days == 1, f"{ec.name}.ttl_days should be 1"

    def test_warm_entities_ttl_is_7_days(self) -> None:
        warm = [ec for ec in ENTITY_REGISTRY.values() if ec.tier == "warm"]
        assert warm, "no warm entities"
        for ec in warm:
            assert ec.ttl_days == 7, f"{ec.name}.ttl_days should be 7"

    def test_cold_entities_ttl_is_30_days(self) -> None:
        cold = [ec for ec in ENTITY_REGISTRY.values() if ec.tier == "cold"]
        assert cold, "no cold entities"
        for ec in cold:
            assert ec.ttl_days == 30, f"{ec.name}.ttl_days should be 30"

    def test_live_entities_have_no_ttl(self) -> None:
        live = [ec for ec in ENTITY_REGISTRY.values() if ec.tier == "live"]
        assert live, "no live entities"
        for ec in live:
            assert ec.ttl_days is None, f"{ec.name}.ttl_days should be None"

    def test_all_four_origins_represented(self) -> None:
        origins = {ec.origin for ec in ENTITY_REGISTRY.values()}
        assert origins == {"postgres", "mongo", "saas", "docs"}

    def test_is_stale_after_ttl_exceeded(self) -> None:
        ec = ENTITY_REGISTRY["assignments"]  # hot, TTL=1
        ec2 = copy.deepcopy(ec)
        ec2.last_refresh = _T_PRIME
        T_stale = _T_PRIME + timedelta(days=2)
        assert ec2.is_stale_at(T_stale)

    def test_is_not_stale_before_ttl_exceeded(self) -> None:
        ec = copy.deepcopy(ENTITY_REGISTRY["prices"])  # warm, TTL=7
        ec.last_refresh = _T_PRIME
        T_fresh = _T_PRIME + timedelta(days=5)
        assert not ec.is_stale_at(T_fresh)

    def test_never_refreshed_is_always_stale(self) -> None:
        ec = copy.deepcopy(ENTITY_REGISTRY["assignments"])
        ec.last_refresh = None
        assert ec.is_stale_at(_T_PRIME)

    def test_live_entity_is_never_stale(self) -> None:
        ec = copy.deepcopy(ENTITY_REGISTRY["consumption_facts"])
        ec.last_refresh = date(2020, 1, 1)  # very old, but still live
        assert not ec.is_stale_at(date(2099, 1, 1))

    def test_measure_to_entity_covers_all_measures(self) -> None:
        all_measures = {m for ec in ENTITY_REGISTRY.values() for m in ec.measures}
        assert set(MEASURE_TO_ENTITY.keys()) == all_measures

    def test_assignments_and_user_status_are_hot(self) -> None:
        assert ENTITY_REGISTRY["assignments"].tier == "hot"
        assert ENTITY_REGISTRY["user_status"].tier == "hot"

    def test_contract_terms_is_cold(self) -> None:
        assert ENTITY_REGISTRY["contract_terms"].tier == "cold"
        assert ENTITY_REGISTRY["contract_terms"].staged_table is None

    def test_consumption_facts_is_live_postgres(self) -> None:
        ec = ENTITY_REGISTRY["consumption_facts"]
        assert ec.tier == "live"
        assert ec.origin == "postgres"
        assert ec.staged_table is None


# ═════════════════════════════════════════════════════════════════════════════
# 2. ETL: refresh_due scheduling
# ═════════════════════════════════════════════════════════════════════════════


class TestRefreshDue:
    def _refreshed_registry(self, at: date) -> dict[str, EntityClass]:
        reg = copy.deepcopy(ENTITY_REGISTRY)
        for ec in reg.values():
            if ec.tier != "live":
                ec.last_refresh = at
        return reg

    def test_at_2_days_only_hot_entities_due(self) -> None:
        reg = self._refreshed_registry(_T_PRIME)
        T = _T_PRIME + timedelta(days=2)
        due_names = {ec.name for ec in refresh_due(reg, T)}
        # hot TTL=1 → stale after 1 day
        assert "assignments" in due_names
        assert "user_status" in due_names
        # warm TTL=7 → still fresh at day 2
        assert "prices" not in due_names
        assert "cost_center_membership" not in due_names
        # cold TTL=30 → still fresh
        assert "contract_terms" not in due_names

    def test_at_10_days_hot_and_warm_due(self) -> None:
        reg = self._refreshed_registry(_T_PRIME)
        T = _T_PRIME + timedelta(days=10)
        due_names = {ec.name for ec in refresh_due(reg, T)}
        assert "assignments" in due_names
        assert "user_status" in due_names
        assert "prices" in due_names
        assert "cost_center_membership" in due_names
        # cold TTL=30 → still fresh at day 10
        assert "contract_terms" not in due_names

    def test_at_40_days_all_cacheable_due(self) -> None:
        reg = self._refreshed_registry(_T_PRIME)
        T = _T_PRIME + timedelta(days=40)
        due_names = {ec.name for ec in refresh_due(reg, T)}
        assert "assignments" in due_names
        assert "prices" in due_names
        assert "contract_terms" in due_names
        assert "vendor_dims" in due_names

    def test_live_entities_never_in_refresh_due(self) -> None:
        reg = self._refreshed_registry(_T_PRIME)
        T = _T_PRIME + timedelta(days=999)
        due_names = {ec.name for ec in refresh_due(reg, T)}
        assert "consumption_facts" not in due_names
        assert "utilization_current" not in due_names
        assert "tickets" not in due_names

    def test_staged_sqlite_setup_creates_schema(self) -> None:
        engine = create_engine("sqlite:///:memory:", future=True)
        create_staged_schema(engine)
        with engine.connect() as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                ).fetchall()
            }
        expected = {
            "staged_users",
            "staged_assignments",
            "staged_license_purchases",
            "staged_cost_centers",
            "staged_vendors",
        }
        assert expected == tables

    def test_setup_stamps_last_refresh(self) -> None:
        reg = copy.deepcopy(ENTITY_REGISTRY)
        engine = create_engine("sqlite:///:memory:", future=True)
        # setup() with no real connections silently no-ops the data pulls
        # but still stamps last_refresh
        setup(reg, engine, T_prime=_T_PRIME)
        for name, ec in reg.items():
            if ec.tier != "live":
                assert ec.last_refresh == _T_PRIME, f"{name}.last_refresh not stamped"
            else:
                assert ec.last_refresh is None, f"{name} (live) should not have last_refresh"

    def test_refresh_entity_mongo_populates_staged_users(self) -> None:
        from churnbench.arms.grounding.etl import refresh_entity

        mongo_db: MagicMock = MagicMock()
        mongo_db["users"].find.return_value = [
            {
                "user_ext_id": "u1",
                "cost_center_id": "cc_000",
                "active": True,
                "hired_at": "2020-01-01",
            },
            {
                "user_ext_id": "u2",
                "cost_center_id": "cc_001",
                "active": False,
                "hired_at": "2021-03-15",
            },
        ]
        engine = create_engine("sqlite:///:memory:", future=True)
        create_staged_schema(engine)
        refresh_entity("user_status", engine, mongo_db=mongo_db, T_prime=_T_PRIME)
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM staged_users")).scalar()
        assert count == 2

    def test_refresh_entity_mongo_sets_active_flag(self) -> None:
        from churnbench.arms.grounding.etl import refresh_entity

        mongo_db: MagicMock = MagicMock()
        mongo_db["users"].find.return_value = [
            {"user_ext_id": "u1", "cost_center_id": "cc_000", "active": True},
            {"user_ext_id": "u2", "cost_center_id": "cc_000", "active": False},
        ]
        engine = create_engine("sqlite:///:memory:", future=True)
        create_staged_schema(engine)
        refresh_entity("user_status", engine, mongo_db=mongo_db, T_prime=_T_PRIME)
        with engine.connect() as conn:
            active = conn.execute(
                text("SELECT COUNT(*) FROM staged_users WHERE active = 1")
            ).scalar()
        assert active == 1


# ═════════════════════════════════════════════════════════════════════════════
# 3. Router — table-driven tests
# ═════════════════════════════════════════════════════════════════════════════


class TestRouter:
    def _registry_fresh(self) -> dict[str, EntityClass]:
        """Registry with all entities refreshed at T_PRIME."""
        reg = copy.deepcopy(ENTITY_REGISTRY)
        for ec in reg.values():
            if ec.tier != "live":
                ec.last_refresh = _T_PRIME
        return reg

    def _registry_stale_hot(self) -> dict[str, EntityClass]:
        """Registry where hot entities are stale (refreshed 5 days ago)."""
        reg = copy.deepcopy(ENTITY_REGISTRY)
        for ec in reg.values():
            if ec.tier != "live":
                ec.last_refresh = _T_PRIME - timedelta(days=5)
        return reg

    def test_fresh_hot_entity_routes_to_staged_sql(self) -> None:
        reg = self._registry_fresh()
        T = _T_PRIME + timedelta(hours=6)  # well within TTL=1 day
        d = decide("user_status", "active_user_count_cc", T, {}, reg)
        assert d.route == "staged_sql"
        assert d.query_method == "templated"
        assert d.cache_miss_reason is None

    def test_stale_hot_mongo_entity_routes_to_origin_live_mongo(self) -> None:
        reg = self._registry_stale_hot()
        T = _T_PRIME  # 5 days after last_refresh → hot TTL exceeded
        d = decide("user_status", "active_user_count_cc", T, {}, reg)
        assert d.route == "origin_live_mongo"
        assert d.cache_miss_reason == "ttl_expired"

    def test_stale_entity_has_cache_miss_reason(self) -> None:
        reg = copy.deepcopy(ENTITY_REGISTRY)
        reg["assignments"].last_refresh = date(2020, 1, 1)  # very stale
        d = decide("assignments", "idle_license_count_cc", _T_EVAL, {}, reg)
        assert d.cache_miss_reason == "ttl_expired"

    def test_never_refreshed_entity_has_never_refreshed_reason(self) -> None:
        reg = copy.deepcopy(ENTITY_REGISTRY)
        # last_refresh remains None
        d = decide("user_status", "active_user_count_cc", _T_EVAL, {}, reg)
        assert d.cache_miss_reason == "never_refreshed"

    def test_consumption_facts_routes_to_warehouse_live(self) -> None:
        reg = self._registry_fresh()
        d = decide("consumption_facts", "total_session_minutes", _T_EVAL, {}, reg)
        assert d.route == "warehouse_live"
        assert d.cache_miss_reason is None

    def test_warehouse_known_measure_is_templated(self) -> None:
        reg = self._registry_fresh()
        d = decide("consumption_facts", "total_session_minutes", _T_EVAL, {}, reg)
        assert d.query_method == "templated"
        assert d.sql_template is not None

    def test_contract_terms_routes_to_docs_index(self) -> None:
        reg = self._registry_fresh()
        d = decide("contract_terms", "contract_annual_value", _T_EVAL, {}, reg)
        assert d.route == "docs_index"
        assert d.query_method == "vector_search"

    def test_no_freshness_tiers_routes_stale_hot_to_staged_sql(self) -> None:
        reg = self._registry_stale_hot()
        T = _T_PRIME  # stale
        d = decide("user_status", "active_user_count_cc", T, {}, reg, no_freshness_tiers=True)
        # With no_freshness_tiers, staleness is ignored → still staged_sql
        assert d.route == "staged_sql"
        assert d.cache_miss_reason is None

    def test_no_source_routing_always_returns_docs_index(self) -> None:
        reg = self._registry_fresh()
        for ec_name in ["user_status", "assignments", "prices", "consumption_facts"]:
            d = decide(ec_name, None, _T_EVAL, {}, reg, no_source_routing=True)
            assert d.route == "docs_index", f"{ec_name} should route to docs_index"
            assert d.query_method == "vector_search"

    def test_unknown_entity_falls_back_to_docs_index(self) -> None:
        reg = self._registry_fresh()
        d = decide("nonexistent_entity", None, _T_EVAL, {}, reg)
        assert d.route == "docs_index"
        assert d.cache_miss_reason == "unknown_entity"

    def test_saas_live_entity_routes_to_origin_live_saas(self) -> None:
        reg = self._registry_fresh()
        d = decide("utilization_current", "current_utilization_product", _T_EVAL, {}, reg)
        assert d.route == "origin_live_saas"
        assert d.query_method == "api_call"

    def test_all_staged_templates_are_known_measures(self) -> None:
        all_measures = set(MEASURE_TO_ENTITY.keys())
        for m in STAGED_SQL_TEMPLATES:
            assert m in all_measures, f"template measure '{m}' not in semantic model"

    def test_all_warehouse_templates_are_live_measures(self) -> None:
        live_measures = {
            m for name, ec in ENTITY_REGISTRY.items() if ec.tier == "live" for m in ec.measures
        }
        for m in WAREHOUSE_SQL_TEMPLATES:
            assert m in live_measures, f"warehouse template '{m}' not in live measures"

    def test_keyword_entity_classes_contract_question(self) -> None:
        q = "What is the contract renewal date for vendor ACME?"
        result = keyword_entity_classes(q)
        assert "contract_terms" in result

    def test_keyword_entity_classes_spend_question(self) -> None:
        q = "What is the monthly spend for cost center cc_000?"
        result = keyword_entity_classes(q)
        assert "prices" in result

    def test_keyword_entity_classes_fallback(self) -> None:
        q = "zzz_completely_unrelated_gibberish"
        result = keyword_entity_classes(q)
        assert result == ["user_status"]  # default fallback


# ═════════════════════════════════════════════════════════════════════════════
# 4. Two-LLM-call flow
# ═════════════════════════════════════════════════════════════════════════════


class TestTwoCallFlow:
    def _arm_with_populated_db(self, lm: _FakeLM) -> GroundingArm:
        """Arm with 3 active users in cc_000 in the staged store."""
        users = [
            {"user_id": "u1", "cost_center_id": "cc_000", "active": True},
            {"user_id": "u2", "cost_center_id": "cc_000", "active": True},
            {"user_id": "u3", "cost_center_id": "cc_000", "active": False},
        ]
        engine = _make_staged_engine(users)
        return _minimal_arm(lm=lm, staged_engine=engine)

    def test_happy_path_exactly_two_lm_calls(self) -> None:
        need_json = json.dumps(
            {
                "entity_classes": ["user_status"],
                "measures": ["active_user_count_cc"],
                "filters": {"cost_center": "cc_000"},
            }
        )
        # Scripted: call 1 = need-resolution, call 2 = synthesis (after staged SQL)
        lm = _FakeLM(
            [
                _ai_message(need_json, input_tokens=80, output_tokens=30),
                _ai_message("2", input_tokens=60, output_tokens=5),
            ]
        )
        arm = self._arm_with_populated_db(lm)
        task = _make_task()
        result = arm.answer(task)
        # FakeLM iterator should be exhausted — exactly 2 calls consumed
        with pytest.raises(StopIteration):
            lm._resp.__next__()  # type: ignore[attr-defined]
        assert result.answer_parsed == 2

    def test_trace_has_need_resolution_role(self) -> None:
        need_json = json.dumps(
            {
                "entity_classes": ["user_status"],
                "measures": ["active_user_count_cc"],
                "filters": {"cost_center": "cc_000"},
            }
        )
        lm = _FakeLM(
            [
                _ai_message(need_json, input_tokens=80, output_tokens=30),
                _ai_message("2", input_tokens=60, output_tokens=5),
            ]
        )
        arm = self._arm_with_populated_db(lm)
        result = arm.answer(_make_task())
        roles = [e["role"] for e in result.trace]
        assert "need_resolution" in roles

    def test_trace_has_retrieval_role(self) -> None:
        need_json = json.dumps(
            {
                "entity_classes": ["user_status"],
                "measures": ["active_user_count_cc"],
                "filters": {"cost_center": "cc_000"},
            }
        )
        lm = _FakeLM(
            [
                _ai_message(need_json, input_tokens=80, output_tokens=30),
                _ai_message("2", input_tokens=60, output_tokens=5),
            ]
        )
        arm = self._arm_with_populated_db(lm)
        result = arm.answer(_make_task())
        retrieval_entries = [e for e in result.trace if e["role"] == "retrieval"]
        assert len(retrieval_entries) >= 1

    def test_trace_has_synthesis_role(self) -> None:
        need_json = json.dumps(
            {
                "entity_classes": ["user_status"],
                "measures": ["active_user_count_cc"],
                "filters": {"cost_center": "cc_000"},
            }
        )
        lm = _FakeLM(
            [
                _ai_message(need_json, input_tokens=80, output_tokens=30),
                _ai_message("2", input_tokens=60, output_tokens=5),
            ]
        )
        arm = self._arm_with_populated_db(lm)
        result = arm.answer(_make_task())
        roles = [e["role"] for e in result.trace]
        assert "synthesis" in roles

    def test_templated_measure_uses_staged_sql_route(self) -> None:
        """active_user_count_cc is in STAGED_SQL_TEMPLATES → staged_sql, no extra LLM call."""
        need_json = json.dumps(
            {
                "entity_classes": ["user_status"],
                "measures": ["active_user_count_cc"],
                "filters": {"cost_center": "cc_000"},
            }
        )
        lm = _FakeLM(
            [
                _ai_message(need_json, input_tokens=80, output_tokens=30),
                _ai_message("2", input_tokens=60, output_tokens=5),
            ]
        )
        arm = self._arm_with_populated_db(lm)
        result = arm.answer(_make_task())
        retrieval = next(e for e in result.trace if e["role"] == "retrieval")
        assert retrieval["route"] == "staged_sql"
        assert retrieval["query_method"] == "templated"

    def test_answer_raw_returned_correctly(self) -> None:
        need_json = json.dumps(
            {
                "entity_classes": ["user_status"],
                "measures": ["active_user_count_cc"],
                "filters": {"cost_center": "cc_000"},
            }
        )
        lm = _FakeLM(
            [
                _ai_message(need_json, input_tokens=80, output_tokens=30),
                _ai_message("42", input_tokens=60, output_tokens=5),
            ]
        )
        arm = self._arm_with_populated_db(lm)
        result = arm.answer(_make_task())
        assert result.answer_raw == "42"
        assert result.answer_parsed == 42

    def test_parse_need_json_extracts_dict(self) -> None:
        raw = '{"entity_classes": ["user_status"], "measures": ["active_user_count_cc"], "filters": {}}'
        result = _parse_need_json(raw)
        assert result["entity_classes"] == ["user_status"]
        assert result["measures"] == ["active_user_count_cc"]

    def test_parse_need_json_handles_markdown_fences(self) -> None:
        raw = 'Here is the JSON:\n```json\n{"entity_classes": ["prices"], "measures": [], "filters": {}}\n```'
        result = _parse_need_json(raw)
        assert result["entity_classes"] == ["prices"]

    def test_parse_need_json_returns_empty_on_failure(self) -> None:
        result = _parse_need_json("not valid json at all")
        assert result == {"entity_classes": [], "measures": [], "filters": {}}


# ═════════════════════════════════════════════════════════════════════════════
# 5. Ablation flags
# ═════════════════════════════════════════════════════════════════════════════


class TestAblations:
    def test_no_freshness_tiers_routes_stale_to_staged_sql(self) -> None:
        """no_freshness_tiers disables TTL check; stale cache still serves staged_sql."""
        reg = copy.deepcopy(ENTITY_REGISTRY)
        # last_refresh very old
        reg["user_status"].last_refresh = date(2020, 1, 1)
        d = decide(
            "user_status",
            "active_user_count_cc",
            _T_EVAL,
            {},
            reg,
            no_freshness_tiers=True,
        )
        assert d.route == "staged_sql"
        assert d.cache_miss_reason is None

    def test_no_freshness_tiers_does_not_affect_live_entities(self) -> None:
        """Live entities are always live regardless of no_freshness_tiers."""
        reg = copy.deepcopy(ENTITY_REGISTRY)
        d = decide(
            "consumption_facts",
            "total_api_calls",
            _T_EVAL,
            {},
            reg,
            no_freshness_tiers=True,
        )
        assert d.route == "warehouse_live"

    def test_no_semantic_model_skips_lm_call_for_need_resolution(self) -> None:
        """With no_semantic_model, need-resolution uses keyword heuristics (0 LLM calls)."""
        # Only 1 scripted response: synthesis (no need-resolution LLM call)
        lm = _FakeLM([_ai_message("5", input_tokens=60, output_tokens=5)])
        arm = _minimal_arm(lm=lm)
        arm.no_semantic_model = True
        task = _make_task(question="How many active users are in cc_000?")
        result = arm.answer(task)
        # FakeLM exhausted after exactly 1 call (synthesis only)
        with pytest.raises(StopIteration):
            lm._resp.__next__()  # type: ignore[attr-defined]
        # need_resolution trace entry should report 0 tokens
        nr_entry = next(e for e in result.trace if e["role"] == "need_resolution")
        assert nr_entry["input_tokens"] == 0
        assert nr_entry["output_tokens"] == 0

    def test_no_semantic_model_keyword_identifies_user_entity(self) -> None:
        """Keyword heuristic for 'active users' question should find user_status."""
        arm = _minimal_arm()
        arm.no_semantic_model = True
        need, nr_inp, nr_out = arm._resolve_needs("How many active users are in cc_000?")
        assert "user_status" in need["entity_classes"]
        assert nr_inp == 0 and nr_out == 0

    def test_no_source_routing_all_decisions_are_docs_index(self) -> None:
        """With no_source_routing, router always returns docs_index."""
        reg = copy.deepcopy(ENTITY_REGISTRY)
        for ec in reg.values():
            if ec.tier != "live":
                ec.last_refresh = _T_PRIME
        for ec_name in ["user_status", "assignments", "prices", "consumption_facts"]:
            d = decide(ec_name, None, _T_EVAL, {}, reg, no_source_routing=True)
            assert d.route == "docs_index", f"{ec_name} should route to docs_index"

    def test_no_source_routing_arm_uses_full_coll(self) -> None:
        """Arm with no_source_routing should use _full_coll in _run_docs_index."""
        import chromadb

        chroma = chromadb.EphemeralClient()
        full_coll = chroma.create_collection(name="test_full_nr")

        arm = _minimal_arm()
        arm.no_source_routing = True
        arm._full_coll = full_coll
        arm._docs_coll = None  # docs_coll absent; full_coll should be used

        from churnbench.arms.grounding.router import RouteDecision

        decision = RouteDecision(
            route="docs_index",
            entity_class="user_status",
            measure="active_user_count_cc",
            sql_template=None,
            sql_params=[],
            query_method="vector_search",
            last_refresh=None,
            cache_miss_reason=None,
        )
        result = arm._run_docs_index(decision)
        # Empty collection → "empty index" message (not an exception)
        assert "empty index" in result or "docs_index" in result

    def test_ablation_flags_are_independent(self) -> None:
        """Verify that one flag does not bleed into another flag's domain."""
        # no_freshness_tiers should not change routing for live entities
        reg = copy.deepcopy(ENTITY_REGISTRY)
        for ec in reg.values():
            if ec.tier != "live":
                ec.last_refresh = _T_PRIME
        d_live = decide(
            "consumption_facts",
            "total_api_calls",
            _T_EVAL,
            {},
            reg,
            no_freshness_tiers=True,
            no_source_routing=False,
        )
        # live entity must still go to warehouse_live (no_freshness_tiers irrelevant here)
        assert d_live.route == "warehouse_live"


# ═════════════════════════════════════════════════════════════════════════════
# 6. Trace completeness
# ═════════════════════════════════════════════════════════════════════════════


class TestTraceCompleteness:
    def _run_arm(self, measure: str = "active_user_count_cc") -> Any:
        need_json = json.dumps(
            {
                "entity_classes": ["user_status"],
                "measures": [measure],
                "filters": {"cost_center": "cc_000"},
            }
        )
        lm = _FakeLM(
            [
                _ai_message(need_json, input_tokens=80, output_tokens=30),
                _ai_message("3", input_tokens=60, output_tokens=5),
            ]
        )
        arm = _minimal_arm(lm=lm)
        return arm.answer(_make_task())

    def test_retrieval_entries_have_route(self) -> None:
        result = self._run_arm()
        for entry in result.trace:
            if entry["role"] == "retrieval":
                assert "route" in entry and entry["route"]

    def test_retrieval_entries_have_last_refresh(self) -> None:
        result = self._run_arm()
        for entry in result.trace:
            if entry["role"] == "retrieval":
                # last_refresh may be None (live entities) or an ISO date string
                assert "last_refresh" in entry

    def test_retrieval_entries_have_staged_vs_live(self) -> None:
        result = self._run_arm()
        for entry in result.trace:
            if entry["role"] == "retrieval":
                assert entry.get("staged_vs_live") in ("staged", "live")

    def test_retrieval_entries_have_entity_class(self) -> None:
        result = self._run_arm()
        for entry in result.trace:
            if entry["role"] == "retrieval":
                assert "entity_class" in entry and entry["entity_class"]

    def test_retrieval_entries_have_query_method(self) -> None:
        result = self._run_arm()
        for entry in result.trace:
            if entry["role"] == "retrieval":
                assert "query_method" in entry

    def test_staged_entity_has_last_refresh_date(self) -> None:
        result = self._run_arm("active_user_count_cc")
        # user_status is hot-tier staged → last_refresh should be set (not None)
        retrieval = next(e for e in result.trace if e["role"] == "retrieval")
        assert retrieval["last_refresh"] == _T_EVAL.isoformat()

    def test_staged_retrieval_shows_staged(self) -> None:
        result = self._run_arm("active_user_count_cc")
        retrieval = next(e for e in result.trace if e["role"] == "retrieval")
        assert retrieval["staged_vs_live"] == "staged"

    def test_synthesis_entry_has_token_counts(self) -> None:
        result = self._run_arm()
        synth = next(e for e in result.trace if e["role"] == "synthesis")
        assert synth["input_tokens"] == 60
        assert synth["output_tokens"] == 5

    def test_need_resolution_entry_has_entity_classes(self) -> None:
        result = self._run_arm()
        nr = next(e for e in result.trace if e["role"] == "need_resolution")
        assert isinstance(nr["entity_classes"], list)
        assert "user_status" in nr["entity_classes"]


# ═════════════════════════════════════════════════════════════════════════════
# 7. Cost hand-computed
# ═════════════════════════════════════════════════════════════════════════════


class TestCostHandComputed:
    def test_happy_path_cost_equals_two_calls(self) -> None:
        """Happy-path = exactly 2 LLM calls; cost = cost_usd(model, inp1+inp2, out1+out2)."""
        inp1, out1 = 80, 30  # need-resolution
        inp2, out2 = 60, 5  # synthesis

        need_json = json.dumps(
            {
                "entity_classes": ["user_status"],
                "measures": ["active_user_count_cc"],
                "filters": {"cost_center": "cc_000"},
            }
        )
        lm = _FakeLM(
            [
                _ai_message(need_json, input_tokens=inp1, output_tokens=out1),
                _ai_message("3", input_tokens=inp2, output_tokens=out2),
            ]
        )
        arm = _minimal_arm(lm=lm)
        result = arm.answer(_make_task())

        expected = cost_usd("claude-sonnet-4-6", inp1 + inp2, out1 + out2, 0)
        assert result.cost_usd == expected

    def test_no_semantic_model_cost_equals_one_call(self) -> None:
        """no_semantic_model skips need-resolution LLM call → 1 call cost."""
        inp_synth, out_synth = 60, 5

        lm = _FakeLM([_ai_message("3", input_tokens=inp_synth, output_tokens=out_synth)])
        arm = _minimal_arm(lm=lm)
        arm.no_semantic_model = True

        result = arm.answer(_make_task())

        expected = cost_usd("claude-sonnet-4-6", inp_synth, out_synth, 0)
        assert result.cost_usd == expected
