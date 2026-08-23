"""Stage 1 — Federated infrastructure tests.

Verifies that:
  1. FederatedTemplate is correctly defined and callable.
  2. decide() emits route='federated' / query_method='federated_template' for measures
     registered in FEDERATED_TEMPLATES.
  3. _run_federated() dispatches both the staged SQLite query and the live Postgres
     query, calls join_fn with the resulting row dicts, and returns the scalar.
  4. The retrieval trace entry contains staged_row_count and live_row_count.

No actual template implementations (SO1/SO3/UT5) are added here — this file tests
the infrastructure only.  All tests are fully offline: in-memory SQLite for staged,
MagicMock for Postgres.
"""

from __future__ import annotations

import copy
from datetime import date
from typing import Any
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine, text

from churnbench.arms.grounding.arm import GroundingArm
from churnbench.arms.grounding.etl import create_staged_schema
from churnbench.arms.grounding.router import (
    FEDERATED_TEMPLATES,
    FederatedTemplate,
    RouteDecision,
    decide,
)
from churnbench.arms.grounding.semantic_model import ENTITY_REGISTRY

_T_PRIME = date(2024, 1, 1)
_T_EVAL = date(2024, 1, 15)

# ── Shared helpers ────────────────────────────────────────────────────────────


def _make_staged_engine() -> Any:
    engine = create_engine("sqlite:///:memory:", future=True)
    create_staged_schema(engine)
    return engine


def _insert_assignment(engine: Any, user_id: str, product_id: str, lic_id: str) -> None:
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO staged_assignments "
                "(assignment_id, license_id, user_id, product_id, staged_at) "
                "VALUES (:aid, :lic, :uid, :pid, :ts)"
            ),
            {
                "aid": f"{lic_id}__{user_id}",
                "lic": lic_id,
                "uid": user_id,
                "pid": product_id,
                "ts": _T_PRIME.isoformat(),
            },
        )
        conn.commit()


def _minimal_arm(staged_engine: Any | None = None) -> GroundingArm:
    engine = staged_engine or _make_staged_engine()
    arm = GroundingArm(_staged_engine=engine)
    arm._lm = MagicMock()
    arm._model = "claude-sonnet-4-6"
    arm._pg_engine = MagicMock()
    arm._mongo_db = MagicMock()
    arm._saas = MagicMock()
    arm._embed = MagicMock()
    arm._staged_engine = engine
    arm._docs_coll = MagicMock()
    arm._full_coll = None
    arm._registry = copy.deepcopy(ENTITY_REGISTRY)
    for ec in arm._registry.values():
        if ec.tier != "live":
            ec.last_refresh = _T_EVAL
    arm._embedding_tokens = 0
    arm._t_prime = _T_PRIME
    return arm


def _pg_mock_with_rows(cols: list[str], rows: list[tuple[Any, ...]]) -> MagicMock:
    """Postgres engine mock that returns given rows from any execute() call."""
    mock_pg = MagicMock()
    mock_conn = MagicMock()
    mock_conn.execute.return_value.keys.return_value = cols
    mock_conn.execute.return_value.fetchall.return_value = rows
    mock_pg.connect.return_value.__enter__.return_value = mock_conn
    mock_pg.connect.return_value.__exit__.return_value = False
    return mock_pg


def _dummy_template(join_result: Any = 99) -> FederatedTemplate:
    return FederatedTemplate(
        staged_sql="SELECT user_id FROM staged_assignments WHERE product_id = :product_id",
        warehouse_sql="SELECT DISTINCT user_ext_id FROM sam.fact_consumption_event",
        join_fn=lambda staged, live: join_result,
        staged_params=["product_id"],
        warehouse_params=[],
    )


# ── FederatedTemplate dataclass ───────────────────────────────────────────────


class TestFederatedTemplateDataclass:
    def test_instantiation(self) -> None:
        tmpl = _dummy_template(42)
        assert tmpl.staged_sql.startswith("SELECT")
        assert tmpl.warehouse_sql.startswith("SELECT")
        assert tmpl.staged_params == ["product_id"]
        assert tmpl.warehouse_params == []

    def test_join_fn_is_callable(self) -> None:
        tmpl = _dummy_template(7)
        result = tmpl.join_fn([{"user_id": "u1"}], [{"user_ext_id": "u1"}])
        assert result == 7

    def test_join_fn_receives_row_dicts(self) -> None:
        seen: list[Any] = []

        def capture(staged: list[dict], live: list[dict]) -> int:
            seen.extend([staged, live])
            return 0

        tmpl = FederatedTemplate(
            staged_sql="SELECT 1",
            warehouse_sql="SELECT 1",
            join_fn=capture,
            staged_params=[],
            warehouse_params=[],
        )
        tmpl.join_fn([{"a": 1}], [{"b": 2}])
        assert seen[0] == [{"a": 1}]
        assert seen[1] == [{"b": 2}]


# ── decide() routing ──────────────────────────────────────────────────────────


class TestDecideFederatedRoute:
    def test_decide_returns_federated_for_registered_measure(self) -> None:
        tmpl = _dummy_template()
        registry = copy.deepcopy(ENTITY_REGISTRY)
        registry["assignments"].measures.append("_infra_test_measure")
        registry["assignments"].last_refresh = _T_EVAL

        with patch.dict(FEDERATED_TEMPLATES, {"_infra_test_measure": tmpl}):
            d = decide("assignments", "_infra_test_measure", _T_EVAL, {}, registry=registry)

        assert d.route == "federated"
        assert d.query_method == "federated_template"
        assert d.measure == "_infra_test_measure"
        assert d.sql_template is None
        assert d.sql_params == []

    def test_decide_does_not_federate_staged_sql_measure(self) -> None:
        """A measure already in STAGED_SQL_TEMPLATES must NOT be overridden by FEDERATED."""
        tmpl = _dummy_template()
        registry = copy.deepcopy(ENTITY_REGISTRY)
        registry["assignments"].last_refresh = _T_EVAL

        with patch.dict(FEDERATED_TEMPLATES, {"assigned_license_count": tmpl}):
            d = decide("assignments", "assigned_license_count", _T_EVAL, {}, registry=registry)

        assert d.route == "staged_sql"  # staged wins; federated only catches the long tail
        assert d.query_method == "templated"

    def test_decide_falls_through_to_llm_for_unknown_measure(self) -> None:
        registry = copy.deepcopy(ENTITY_REGISTRY)
        registry["assignments"].last_refresh = _T_EVAL
        d = decide("assignments", "completely_unknown_measure", _T_EVAL, {}, registry=registry)
        assert d.route == "staged_sql"
        assert d.query_method == "llm_generated"


# ── _run_federated() + trace ──────────────────────────────────────────────────


class TestRunFederated:
    def test_dispatches_both_queries_and_joins(self) -> None:
        """Both staged and live queries execute; join_fn is called with row dicts."""
        calls: list[tuple[list[dict], list[dict]]] = []

        def join_fn(staged: list[dict], live: list[dict]) -> int:
            calls.append((staged, live))
            return 5

        tmpl = FederatedTemplate(
            staged_sql="SELECT user_id FROM staged_assignments WHERE product_id = :product_id",
            warehouse_sql="SELECT DISTINCT user_ext_id FROM sam.fact_consumption_event",
            join_fn=join_fn,
            staged_params=["product_id"],
            warehouse_params=[],
        )

        engine = _make_staged_engine()
        _insert_assignment(engine, "u1", "prod_001", "lic1")
        _insert_assignment(engine, "u2", "prod_001", "lic2")

        pg = _pg_mock_with_rows(["user_ext_id"], [("u1",)])
        arm = _minimal_arm(staged_engine=engine)
        arm._pg_engine = pg

        decision = RouteDecision(
            route="federated",
            entity_class="assignments",
            measure="_test_disp",
            sql_template=None,
            sql_params=[],
            query_method="federated_template",
            last_refresh=_T_EVAL,
            cache_miss_reason=None,
        )

        with patch.dict(FEDERATED_TEMPLATES, {"_test_disp": tmpl}):
            result_text, trace = arm._execute_one(decision, {"product_id": "prod_001"})

        assert len(calls) == 1
        staged_rows, live_rows = calls[0]
        # Both rows for prod_001 reached the join function
        assert len(staged_rows) == 2
        assert all(r["user_id"] in ("u1", "u2") for r in staged_rows)
        # Live row came through
        assert len(live_rows) == 1
        assert live_rows[0]["user_ext_id"] == "u1"
        # Result text includes the joined scalar
        assert "5" in result_text

    def test_trace_contains_required_fields(self) -> None:
        """Trace entry carries staged_row_count, live_row_count, query_method."""
        engine = _make_staged_engine()
        _insert_assignment(engine, "u1", "prod_001", "lic1")

        pg = _pg_mock_with_rows(["user_ext_id"], [("u99",), ("u100",)])
        arm = _minimal_arm(staged_engine=engine)
        arm._pg_engine = pg

        tmpl = FederatedTemplate(
            staged_sql="SELECT user_id FROM staged_assignments",
            warehouse_sql="SELECT DISTINCT user_ext_id FROM sam.fact_consumption_event",
            join_fn=lambda s, _live: 0,
            staged_params=[],
            warehouse_params=[],
        )
        decision = RouteDecision(
            route="federated",
            entity_class="assignments",
            measure="_test_trace",
            sql_template=None,
            sql_params=[],
            query_method="federated_template",
            last_refresh=_T_EVAL,
            cache_miss_reason=None,
        )

        with patch.dict(FEDERATED_TEMPLATES, {"_test_trace": tmpl}):
            _, trace = arm._execute_one(decision, {})

        assert trace["route"] == "federated"
        assert trace["query_method"] == "federated_template"
        assert trace["staged_row_count"] == 1
        assert trace["live_row_count"] == 2

    def test_staged_error_surfaces_in_result_text(self) -> None:
        """If the staged SQL fails, result_text carries the error and arm does not crash."""
        arm = _minimal_arm()
        # Break the staged engine
        arm._staged_engine = MagicMock()
        arm._staged_engine.connect.side_effect = Exception("sqlite gone")

        tmpl = _dummy_template()
        decision = RouteDecision(
            route="federated",
            entity_class="assignments",
            measure="_err_test",
            sql_template=None,
            sql_params=[],
            query_method="federated_template",
            last_refresh=_T_EVAL,
            cache_miss_reason=None,
        )

        # Provide the required param so the missing-param guard doesn't fire first;
        # the staged engine itself should then raise and surface the error.
        with patch.dict(FEDERATED_TEMPLATES, {"_err_test": tmpl}):
            result_text, trace = arm._execute_one(decision, {"product_id": "prd_test"})

        assert "staged error" in result_text
        assert "staged_row_count" not in trace  # error before extra_trace was populated

    def test_live_error_surfaces_in_result_text(self) -> None:
        """If the live Postgres query fails, result_text carries the error."""
        engine = _make_staged_engine()
        pg = MagicMock()
        pg.connect.side_effect = Exception("pg down")
        arm = _minimal_arm(staged_engine=engine)
        arm._pg_engine = pg

        # No staged_params so the staged query succeeds (empty table, 0 rows — not an error).
        tmpl = FederatedTemplate(
            staged_sql="SELECT user_id FROM staged_assignments",
            warehouse_sql="SELECT DISTINCT user_ext_id FROM sam.fact_consumption_event",
            join_fn=lambda s, _live: 0,
            staged_params=[],
            warehouse_params=[],
        )
        decision = RouteDecision(
            route="federated",
            entity_class="assignments",
            measure="_pg_err_test",
            sql_template=None,
            sql_params=[],
            query_method="federated_template",
            last_refresh=_T_EVAL,
            cache_miss_reason=None,
        )

        with patch.dict(FEDERATED_TEMPLATES, {"_pg_err_test": tmpl}):
            result_text, trace = arm._execute_one(decision, {})

        assert "live error" in result_text

    def test_missing_template_returns_no_template_message(self) -> None:
        """If the measure has no FederatedTemplate entry, a clear message is returned."""
        arm = _minimal_arm()
        decision = RouteDecision(
            route="federated",
            entity_class="assignments",
            measure="nonexistent_measure",
            sql_template=None,
            sql_params=[],
            query_method="federated_template",
            last_refresh=_T_EVAL,
            cache_miss_reason=None,
        )
        result_text, trace = arm._execute_one(decision, {})
        assert "no template" in result_text
