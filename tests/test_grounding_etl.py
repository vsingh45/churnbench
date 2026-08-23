"""Regression tests for churnbench/arms/grounding/etl.py.

Fully offline: SQLite in-memory for the staged store, MagicMock for Postgres
and MongoDB.  Specifically guards the three behaviours that were previously
silent (returns True/False, last_refresh only stamped on success, table
populated when ETL succeeds).
"""
from __future__ import annotations

import copy
from datetime import date
from typing import Any
from unittest.mock import MagicMock

from sqlalchemy import create_engine, text

from churnbench.arms.grounding.etl import create_staged_schema, refresh_entity, setup
from churnbench.arms.grounding.semantic_model import ENTITY_REGISTRY

_T_PRIME = date(2024, 1, 1)

_PURCHASE_ROWS = [
    {
        "purchase_id": "p001",
        "product_id": "prod_001",
        "cost_center_id": "cc_000",
        "seats": 10,
        "unit_price_usd": 99.99,
        "valid_from": "2024-01-01",
        "valid_until": "2025-01-01",
    },
    {
        "purchase_id": "p002",
        "product_id": "prod_002",
        "cost_center_id": "cc_001",
        "seats": 5,
        "unit_price_usd": 49.50,
        "valid_from": "2024-01-01",
        "valid_until": "2025-01-01",
    },
]


def _staged_engine() -> Any:
    engine = create_engine("sqlite:///:memory:", future=True)
    create_staged_schema(engine)
    return engine


def _pg_engine_with_rows(rows: list[dict[str, Any]]) -> MagicMock:
    """Postgres engine mock that returns `rows` from sam.fact_license_purchase."""
    pg_engine = MagicMock()
    mock_conn = MagicMock()
    mock_conn.execute.return_value.mappings.return_value.all.return_value = rows
    pg_engine.connect.return_value.__enter__.return_value = mock_conn
    pg_engine.connect.return_value.__exit__.return_value = False
    return pg_engine


def _pg_engine_that_raises() -> MagicMock:
    """Postgres engine mock that raises on connect — simulates connection refused."""
    pg_engine = MagicMock()
    pg_engine.connect.side_effect = Exception("ECONNREFUSED: could not connect to postgres")
    return pg_engine


def _mongo_db_with_users(n: int = 3) -> MagicMock:
    users = [
        {
            "user_ext_id": f"u{i}",
            "cost_center_id": "cc_000",
            "active": True,
            "hired_at": "2020-01-01",
        }
        for i in range(n)
    ]
    mongo_db = MagicMock()
    mongo_db["users"].find.return_value = users
    mongo_db["assignments"].find.return_value = []
    return mongo_db


# ── refresh_entity ────────────────────────────────────────────────────────────


class TestRefreshEntityPrices:
    def test_returns_true_and_populates_table(self) -> None:
        engine = _staged_engine()
        pg = _pg_engine_with_rows(_PURCHASE_ROWS)
        result = refresh_entity("prices", engine, pg_engine=pg, T_prime=_T_PRIME)
        assert result is True
        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM staged_license_purchases")
            ).scalar()
        assert count == len(_PURCHASE_ROWS)

    def test_returns_false_when_postgres_unreachable(self) -> None:
        engine = _staged_engine()
        pg = _pg_engine_that_raises()
        result = refresh_entity("prices", engine, pg_engine=pg, T_prime=_T_PRIME)
        assert result is False
        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM staged_license_purchases")
            ).scalar()
        assert count == 0

    def test_row_values_stored_accurately(self) -> None:
        engine = _staged_engine()
        pg = _pg_engine_with_rows(_PURCHASE_ROWS[:1])
        refresh_entity("prices", engine, pg_engine=pg, T_prime=_T_PRIME)
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT seats, unit_price_usd FROM staged_license_purchases")
            ).fetchone()
        assert row is not None
        assert row[0] == 10
        assert abs(row[1] - 99.99) < 0.001


# ── setup() last_refresh stamping ─────────────────────────────────────────────


class TestSetupLastRefresh:
    def test_last_refresh_stamped_on_successful_etl(self) -> None:
        registry = copy.deepcopy(ENTITY_REGISTRY)
        engine = _staged_engine()
        pg = _pg_engine_with_rows(_PURCHASE_ROWS)
        mongo = _mongo_db_with_users()
        setup(registry, engine, pg_engine=pg, mongo_db=mongo, T_prime=_T_PRIME)
        assert registry["prices"].last_refresh == _T_PRIME

    def test_last_refresh_not_stamped_when_postgres_fails(self) -> None:
        """Broken Postgres must NOT mark prices as fresh — the router needs to fall through."""
        registry = copy.deepcopy(ENTITY_REGISTRY)
        engine = _staged_engine()
        pg = _pg_engine_that_raises()
        mongo = _mongo_db_with_users()
        setup(registry, engine, pg_engine=pg, mongo_db=mongo, T_prime=_T_PRIME)
        assert registry["prices"].last_refresh is None

    def test_mongo_entities_stamp_even_when_postgres_fails(self) -> None:
        """Mongo-origin entities (user_status, assignments) are independent of Postgres."""
        registry = copy.deepcopy(ENTITY_REGISTRY)
        engine = _staged_engine()
        pg = _pg_engine_that_raises()
        mongo = _mongo_db_with_users()
        setup(registry, engine, pg_engine=pg, mongo_db=mongo, T_prime=_T_PRIME)
        assert registry["user_status"].last_refresh == _T_PRIME
        assert registry["assignments"].last_refresh == _T_PRIME

    def test_vendor_dims_not_stamped_when_postgres_fails(self) -> None:
        registry = copy.deepcopy(ENTITY_REGISTRY)
        engine = _staged_engine()
        pg = _pg_engine_that_raises()
        mongo = _mongo_db_with_users()
        setup(registry, engine, pg_engine=pg, mongo_db=mongo, T_prime=_T_PRIME)
        assert registry["vendor_dims"].last_refresh is None
