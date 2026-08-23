"""Tests for Stage 2 federated templates and pre-flight staged SQL additions.

Fully offline: join functions are called directly with pre-built row lists;
staged SQL templates are exercised against SQLite in-memory.

Coverage:
  - _join_idle_by_product (SO1 idle_license_count)
  - _join_zero_api_by_product (UT4 zero_usage_license_count)
  - _join_idle_by_cc (SO3 idle_license_count_cc)
  - top_spending_cost_center staged template
  - cost_centers_above_threshold staged template
  - FEDERATED_TEMPLATES dict is populated for the three Stage 2 measures
"""

from __future__ import annotations

from sqlalchemy import create_engine, text

from churnbench.arms.grounding.etl import create_staged_schema
from churnbench.arms.grounding.router import (
    FEDERATED_TEMPLATES,
    STAGED_SQL_TEMPLATES,
    _join_idle_by_cc,
    _join_idle_by_product,
    _join_zero_api_by_product,
)


# ── Staged-SQL helpers ────────────────────────────────────────────────────────


def _engine_with_spend_data(
    prices: list[dict],
    users: list[dict],
    assignments: list[dict],
) -> object:
    """Populate all three staged tables needed for holder-attributed spend queries."""
    engine = create_engine("sqlite:///:memory:", future=True)
    create_staged_schema(engine)
    with engine.connect() as conn:
        for r in prices:
            conn.execute(
                text(
                    "INSERT INTO staged_license_purchases "
                    "(purchase_id, product_id, cost_center_id, seats, unit_price_usd, "
                    "valid_from, valid_until, staged_at) "
                    "VALUES (:purchase_id, :product_id, :cost_center_id, :seats, "
                    ":unit_price_usd, :valid_from, :valid_until, :staged_at)"
                ),
                r,
            )
        for u in users:
            conn.execute(
                text(
                    "INSERT INTO staged_users "
                    "(user_id, cost_center_id, active, hired_at, staged_at) "
                    "VALUES (:user_id, :cost_center_id, :active, :hired_at, :staged_at)"
                ),
                u,
            )
        for a in assignments:
            conn.execute(
                text(
                    "INSERT INTO staged_assignments "
                    "(assignment_id, license_id, user_id, product_id, staged_at) "
                    "VALUES (:assignment_id, :license_id, :user_id, :product_id, :staged_at)"
                ),
                a,
            )
        conn.commit()
    return engine


# Prices: prd_001 at $200/seat (bought by cc_000), prd_002 at $100/seat (cc_001),
#         prd_003 at $50/seat (cc_001).
_PRICE_ROWS = [
    {
        "purchase_id": "p1",
        "product_id": "prd_001",
        "cost_center_id": "cc_000",
        "seats": 10,
        "unit_price_usd": 200.0,
        "valid_from": "2024-01-01",
        "valid_until": "2025-01-01",
        "staged_at": "2024-03-29",
    },
    {
        "purchase_id": "p2",
        "product_id": "prd_002",
        "cost_center_id": "cc_001",
        "seats": 5,
        "unit_price_usd": 100.0,
        "valid_from": "2024-01-01",
        "valid_until": "2025-01-01",
        "staged_at": "2024-03-29",
    },
    {
        "purchase_id": "p3",
        "product_id": "prd_003",
        "cost_center_id": "cc_001",
        "seats": 2,
        "unit_price_usd": 50.0,
        "valid_from": "2024-01-01",
        "valid_until": "2025-01-01",
        "staged_at": "2024-03-29",
    },
]

# Users: 10 active in cc_000, 7 active in cc_001.
_USER_ROWS = [
    {
        "user_id": f"u_{cc}_{i:02d}",
        "cost_center_id": cc,
        "active": 1,
        "hired_at": "2023-01-01",
        "staged_at": "2024-03-29",
    }
    for cc, count in [("cc_000", 10), ("cc_001", 7)]
    for i in range(count)
]

# Assignments: cc_000 users hold all 10 prd_001 licenses (10 × $200 = $2000);
# cc_001 users hold 5 prd_002 licenses (5 × $100 = $500) + 2 prd_003 (2 × $50 = $100) = $600.
_ASSIGNMENT_ROWS = (
    [
        {
            "assignment_id": f"a_001_{i}",
            "license_id": f"lic_001_{i}",
            "user_id": f"u_cc_000_{i:02d}",
            "product_id": "prd_001",
            "staged_at": "2024-03-29",
        }
        for i in range(10)
    ]
    + [
        {
            "assignment_id": f"a_002_{i}",
            "license_id": f"lic_002_{i}",
            "user_id": f"u_cc_001_{i:02d}",
            "product_id": "prd_002",
            "staged_at": "2024-03-29",
        }
        for i in range(5)
    ]
    + [
        {
            "assignment_id": f"a_003_{i}",
            "license_id": f"lic_003_{i}",
            "user_id": f"u_cc_001_{(i+5):02d}",
            "product_id": "prd_003",
            "staged_at": "2024-03-29",
        }
        for i in range(2)
    ]
)


# ── top_spending_cost_center ──────────────────────────────────────────────────
# Spend is attributed to the *holder's* CC (not the purchaser's CC), for active users.
# cc_000 holders: 10 × $200 = $2000; cc_001 holders: 5 × $100 + 2 × $50 = $600 → cc_000 wins.


class TestTopSpendingCostCenter:
    def test_returns_highest_spend_cc(self) -> None:
        engine = _engine_with_spend_data(_PRICE_ROWS, _USER_ROWS, _ASSIGNMENT_ROWS)
        sql, params = STAGED_SQL_TEMPLATES["top_spending_cost_center"]
        assert params == []
        with engine.connect() as conn:
            row = conn.execute(text(sql)).fetchone()
        assert row is not None
        # Holder-attributed: cc_000 gets 10 × $200 = $2000; cc_001 gets $600 → cc_000 wins
        assert row[0] == "cc_000"

    def test_negative_single_cc(self) -> None:
        # One user, one assignment, one purchase — cc_000 is only CC.
        engine = _engine_with_spend_data(
            [_PRICE_ROWS[0]],
            [_USER_ROWS[0]],
            [_ASSIGNMENT_ROWS[0]],
        )
        sql, _ = STAGED_SQL_TEMPLATES["top_spending_cost_center"]
        with engine.connect() as conn:
            row = conn.execute(text(sql)).fetchone()
        assert row is not None and row[0] == "cc_000"


# ── cost_centers_above_threshold ──────────────────────────────────────────────


class TestCostCentersAboveThreshold:
    def test_counts_ccs_above_threshold(self) -> None:
        engine = _engine_with_spend_data(_PRICE_ROWS, _USER_ROWS, _ASSIGNMENT_ROWS)
        sql, params = STAGED_SQL_TEMPLATES["cost_centers_above_threshold"]
        assert params == ["threshold"]
        with engine.connect() as conn:
            # threshold=1000: only cc_000 ($2000) passes; cc_001 ($600) does not
            row = conn.execute(text(sql), {"threshold": 1000.0}).fetchone()
        assert row is not None and row[0] == 1

    def test_negative_threshold_excludes_all(self) -> None:
        engine = _engine_with_spend_data(_PRICE_ROWS, _USER_ROWS, _ASSIGNMENT_ROWS)
        sql, _ = STAGED_SQL_TEMPLATES["cost_centers_above_threshold"]
        with engine.connect() as conn:
            row = conn.execute(text(sql), {"threshold": 9999.0}).fetchone()
        assert row is not None and row[0] == 0

    def test_all_ccs_above_low_threshold(self) -> None:
        engine = _engine_with_spend_data(_PRICE_ROWS, _USER_ROWS, _ASSIGNMENT_ROWS)
        sql, _ = STAGED_SQL_TEMPLATES["cost_centers_above_threshold"]
        with engine.connect() as conn:
            row = conn.execute(text(sql), {"threshold": 0.0}).fetchone()
        assert row is not None and row[0] == 2  # cc_000 ($2000) and cc_001 ($600) both > 0


# ── Federated join helpers ────────────────────────────────────────────────────


class TestJoinIdleByProduct:
    """SO1 idle_license_count: count assignments whose holder had no consumption."""

    def test_positive_all_idle(self) -> None:
        staged = [{"license_id": "lic_1", "user_id": "u1"}]
        live: list[dict] = []  # no one used anything
        assert _join_idle_by_product(staged, live) == 1

    def test_positive_mixed(self) -> None:
        staged = [
            {"license_id": "lic_1", "user_id": "u1"},
            {"license_id": "lic_2", "user_id": "u2"},
            {"license_id": "lic_3", "user_id": "u3"},
        ]
        live = [{"user_ext_id": "u2"}]  # u2 was active
        assert _join_idle_by_product(staged, live) == 2  # u1 and u3 idle

    def test_negative_no_idle(self) -> None:
        staged = [{"license_id": "lic_1", "user_id": "u1"}]
        live = [{"user_ext_id": "u1"}]  # u1 was active
        assert _join_idle_by_product(staged, live) == 0

    def test_negative_empty_staged(self) -> None:
        assert _join_idle_by_product([], []) == 0


class TestJoinZeroApiByProduct:
    """UT4 zero_usage_license_count: count assignments where holder had no api_calls."""

    def test_positive_zero_api(self) -> None:
        staged = [{"license_id": "lic_1", "user_id": "u1"}]
        live: list[dict] = []  # live = users with >0 api_calls; empty means all zero
        assert _join_zero_api_by_product(staged, live) == 1

    def test_negative_has_api_calls(self) -> None:
        staged = [{"license_id": "lic_1", "user_id": "u1"}]
        live = [{"user_ext_id": "u1"}]
        assert _join_zero_api_by_product(staged, live) == 0

    def test_mixed(self) -> None:
        staged = [
            {"license_id": "lic_1", "user_id": "u1"},
            {"license_id": "lic_2", "user_id": "u2"},
        ]
        live = [{"user_ext_id": "u1"}]  # only u1 had api_calls
        assert _join_zero_api_by_product(staged, live) == 1


class TestJoinIdleByCc:
    """SO3 idle_license_count_cc: count (user, product) pairs with no consumption."""

    def test_positive_all_idle(self) -> None:
        staged = [{"license_id": "lic_1", "user_id": "u1", "product_id": "prd_001"}]
        live: list[dict] = []
        assert _join_idle_by_cc(staged, live) == 1

    def test_negative_user_product_active(self) -> None:
        staged = [{"license_id": "lic_1", "user_id": "u1", "product_id": "prd_001"}]
        live = [{"user_ext_id": "u1", "product_sku": "prd_001"}]
        assert _join_idle_by_cc(staged, live) == 0

    def test_different_product_still_idle(self) -> None:
        staged = [{"license_id": "lic_1", "user_id": "u1", "product_id": "prd_001"}]
        # u1 used prd_002, not prd_001 — license for prd_001 is still idle
        live = [{"user_ext_id": "u1", "product_sku": "prd_002"}]
        assert _join_idle_by_cc(staged, live) == 1

    def test_mixed_assignments(self) -> None:
        staged = [
            {"license_id": "lic_1", "user_id": "u1", "product_id": "prd_001"},
            {"license_id": "lic_2", "user_id": "u1", "product_id": "prd_002"},
            {"license_id": "lic_3", "user_id": "u2", "product_id": "prd_001"},
        ]
        live = [
            {"user_ext_id": "u1", "product_sku": "prd_001"},  # u1/prd_001 active
        ]
        # u1/prd_002 idle, u2/prd_001 idle → 2 idle
        assert _join_idle_by_cc(staged, live) == 2


# ── FEDERATED_TEMPLATES dict populated ───────────────────────────────────────


class TestFederatedTemplatesPopulated:
    def test_idle_license_count_defined(self) -> None:
        assert "idle_license_count" in FEDERATED_TEMPLATES
        tmpl = FEDERATED_TEMPLATES["idle_license_count"]
        assert tmpl.staged_params == ["product_id"]
        assert "cutoff_date" in tmpl.warehouse_params

    def test_zero_usage_license_count_defined(self) -> None:
        assert "zero_usage_license_count" in FEDERATED_TEMPLATES
        tmpl = FEDERATED_TEMPLATES["zero_usage_license_count"]
        assert tmpl.staged_params == ["product_id"]
        assert "api_calls > 0" in tmpl.warehouse_sql

    def test_idle_license_count_cc_defined(self) -> None:
        assert "idle_license_count_cc" in FEDERATED_TEMPLATES
        tmpl = FEDERATED_TEMPLATES["idle_license_count_cc"]
        assert tmpl.staged_params == ["cost_center"]
        assert "cutoff_date" in tmpl.warehouse_params
