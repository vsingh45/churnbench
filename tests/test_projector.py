"""Tests for churnbench.fabric.projector.

These tests operate entirely in memory (no Postgres, no MongoDB, no filesystem
writes beyond a tmp directory for docs).  The Projector's _fold_events logic and
project_docs are exercised directly; project_postgres and project_mongo are NOT
called here because they need live Docker services — they belong in integration tests.

Fixtures build a small, hand-crafted ledger so the expected outcomes are readable
and verifiable without running the full 180-day simulator.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from churnbench.ledger.ledger import EventKind, Ledger
from churnbench.ledger.fold import fold_events
from churnbench.fabric.projector import Projector


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _minimal_ledger() -> Ledger:
    """A small deterministic ledger with two products, two users, two contracts.

    Timeline:
      Day 0 (2024-01-01): product prices set, contracts signed, users hired,
                           licenses purchased and assigned.
      Day 5 (2024-01-06): usr_000001 offboarded, license reassigned to usr_000000.
      Day 10 (2024-01-11): contract ctr_0001 renewed, new price for prd_0001.
    """
    led = Ledger()
    d0 = date(2024, 1, 1)
    d5 = date(2024, 1, 6)
    d10 = date(2024, 1, 11)

    # Products (initial prices)
    led.append(
        d0,
        EventKind.PRICE_CHANGED,
        "product",
        "prd_0000",
        {"unit_price_usd": 100.0, "reason": "initial"},
    )
    led.append(
        d0,
        EventKind.PRICE_CHANGED,
        "product",
        "prd_0001",
        {"unit_price_usd": 200.0, "reason": "initial"},
    )

    # Contracts
    led.append(
        d0, EventKind.CONTRACT_SIGNED, "contract", "ctr_0000", {"vendor_idx": 0, "term_months": 12}
    )
    led.append(
        d0, EventKind.CONTRACT_SIGNED, "contract", "ctr_0001", {"vendor_idx": 1, "term_months": 12}
    )

    # Users
    led.append(d0, EventKind.USER_HIRED, "user", "usr_000000", {"cost_center_id": "cc_000"})
    led.append(d0, EventKind.USER_HIRED, "user", "usr_000001", {"cost_center_id": "cc_001"})

    # Licenses purchased and assigned
    led.append(
        d0,
        EventKind.LICENSE_PURCHASED,
        "license",
        "lic_000000",
        {"product_id": "prd_0000", "seats": 1, "unit_price_usd": 100.0},
    )
    led.append(
        d0,
        EventKind.LICENSE_ASSIGNED,
        "license",
        "lic_000000",
        {"to": "usr_000000", "product_id": "prd_0000"},
    )

    led.append(
        d0,
        EventKind.LICENSE_PURCHASED,
        "license",
        "lic_000001",
        {"product_id": "prd_0001", "seats": 1, "unit_price_usd": 200.0},
    )
    led.append(
        d0,
        EventKind.LICENSE_ASSIGNED,
        "license",
        "lic_000001",
        {"to": "usr_000001", "product_id": "prd_0001"},
    )

    # Day 5: offboard usr_000001, unassign then reassign license
    led.append(
        d5,
        EventKind.LICENSE_UNASSIGNED,
        "license",
        "lic_000001",
        {"prev_holder": "usr_000001", "reason": "offboard"},
    )
    led.append(d5, EventKind.USER_OFFBOARDED, "user", "usr_000001", {})
    led.append(
        d5,
        EventKind.LICENSE_REASSIGNED,
        "license",
        "lic_000001",
        {"from": "usr_000001", "to": "usr_000000"},
    )

    # Day 10: contract renewal, price change
    led.append(d10, EventKind.CONTRACT_RENEWED, "contract", "ctr_0001", {"term_months": 24})
    led.append(
        d10,
        EventKind.PRICE_CHANGED,
        "product",
        "prd_0001",
        {"unit_price_usd": 210.0, "prev_price": 200.0, "pct_change": 0.05},
    )

    return led


# ─────────────────────────────────────────────────────────────────────────────
# Test (a): projecting the same ledger at the same T twice yields identical
#           world-state (idempotency of _fold_events, not of DB writes)
# ─────────────────────────────────────────────────────────────────────────────


class TestIdempotency:
    def test_fold_same_T_twice_identical_user_counts(self) -> None:
        led = _minimal_ledger()
        T = date(2024, 1, 20)
        ws1 = fold_events(led.events_through(T))
        ws2 = fold_events(led.events_through(T))
        assert len(ws1.users) == len(ws2.users)

    def test_fold_same_T_twice_identical_license_state(self) -> None:
        led = _minimal_ledger()
        T = date(2024, 1, 20)
        ws1 = fold_events(led.events_through(T))
        ws2 = fold_events(led.events_through(T))
        # Same holder for every license
        for lic_id in ws1.licenses:
            assert ws1.licenses[lic_id].holder_id == ws2.licenses[lic_id].holder_id

    def test_fold_same_T_twice_identical_product_prices(self) -> None:
        led = _minimal_ledger()
        T = date(2024, 1, 20)
        ws1 = fold_events(led.events_through(T))
        ws2 = fold_events(led.events_through(T))
        for pid in ws1.products:
            assert ws1.products[pid].current_price == ws2.products[pid].current_price

    def test_docs_projection_idempotent(self, tmp_path: Path) -> None:
        """Writing docs twice to the same directory produces the same files."""
        led = _minimal_ledger()
        T = date(2024, 1, 20)
        proj = Projector()
        n1 = proj.project_docs(led, T, tmp_path / "docs")
        # Read content from first run
        docs_dir = tmp_path / "docs"
        contents_1 = {p.name: p.read_text() for p in docs_dir.iterdir()}

        n2 = proj.project_docs(led, T, docs_dir)
        contents_2 = {p.name: p.read_text() for p in docs_dir.iterdir()}

        assert n1 == n2
        assert contents_1 == contents_2


# ─────────────────────────────────────────────────────────────────────────────
# Test (b): projecting at T2 > T1 yields >= rows of T1
# ─────────────────────────────────────────────────────────────────────────────


class TestMonotonicity:
    def test_more_events_at_later_T_users(self) -> None:
        """T1=day4 has 2 users; T2=day6 has 1 active (offboard on day5)."""
        led = _minimal_ledger()
        T1 = date(2024, 1, 4)
        T2 = date(2024, 1, 20)
        ws1 = fold_events(led.events_through(T1))
        ws2 = fold_events(led.events_through(T2))
        # More events overall at T2 — total users in dict (active+inactive) can only grow
        assert len(ws2.users) >= len(ws1.users)

    def test_more_events_at_later_T_products(self) -> None:
        led = _minimal_ledger()
        T1 = date(2024, 1, 1)
        T2 = date(2024, 1, 20)
        ws1 = fold_events(led.events_through(T1))
        ws2 = fold_events(led.events_through(T2))
        assert len(ws2.products) >= len(ws1.products)

    def test_more_events_at_later_T_contracts(self) -> None:
        led = _minimal_ledger()
        T1 = date(2024, 1, 1)
        T2 = date(2024, 1, 20)
        ws1 = fold_events(led.events_through(T1))
        ws2 = fold_events(led.events_through(T2))
        assert len(ws2.contracts) >= len(ws1.contracts)

    def test_later_T_has_updated_price(self) -> None:
        """prd_0001 price changes from 200 to 210 on day 10."""
        led = _minimal_ledger()
        ws_before = fold_events(led.events_through(date(2024, 1, 9)))
        ws_after = fold_events(led.events_through(date(2024, 1, 11)))
        assert ws_before.products["prd_0001"].current_price == pytest.approx(200.0)
        assert ws_after.products["prd_0001"].current_price == pytest.approx(210.0)

    def test_docs_count_monotone(self, tmp_path: Path) -> None:
        """Contract count at T2 >= T1 (contracts can only be added, never deleted)."""
        led = _minimal_ledger()
        proj = Projector()
        # Before any contracts (impossible here but test the general principle)
        n1 = proj.project_docs(led, date(2024, 1, 1), tmp_path / "docs_t1")
        n2 = proj.project_docs(led, date(2024, 1, 20), tmp_path / "docs_t2")
        assert n2 >= n1


# ─────────────────────────────────────────────────────────────────────────────
# Test (c): USER_OFFBOARDED removes that user from Mongo users
# ─────────────────────────────────────────────────────────────────────────────


class TestOffboardSemantics:
    def test_offboarded_user_not_active_at_T_after_event(self) -> None:
        """usr_000001 is offboarded on day 5; any T >= day5 must show them inactive."""
        led = _minimal_ledger()
        T = date(2024, 1, 10)
        ws = fold_events(led.events_through(T))
        assert "usr_000001" in ws.users
        assert ws.users["usr_000001"].active is False

    def test_offboarded_user_active_before_event(self) -> None:
        """At T=day4 (before offboard on day5), usr_000001 must still be active."""
        led = _minimal_ledger()
        T = date(2024, 1, 4)
        ws = fold_events(led.events_through(T))
        assert ws.users["usr_000001"].active is True

    def test_offboarded_user_excluded_from_mongo_users_collection(self) -> None:
        """Simulate what project_mongo would write: only active users in 'users' coll."""
        led = _minimal_ledger()
        T = date(2024, 1, 10)
        ws = fold_events(led.events_through(T))
        # This mirrors exactly what _mongo_users() does
        active_ids = {u.user_id for u in ws.users.values() if u.active}
        assert "usr_000001" not in active_ids
        assert "usr_000000" in active_ids

    def test_license_reassigned_after_offboard(self) -> None:
        """lic_000001 was held by usr_000001; after offboard it should be with usr_000000."""
        led = _minimal_ledger()
        T = date(2024, 1, 10)
        ws = fold_events(led.events_through(T))
        assert ws.licenses["lic_000001"].holder_id == "usr_000000"

    def test_offboarded_user_not_in_entitlements(self) -> None:
        """Entitlement grouping must skip inactive users."""
        from collections import defaultdict

        led = _minimal_ledger()
        T = date(2024, 1, 10)
        ws = fold_events(led.events_through(T))
        by_user: dict[str, list[str]] = defaultdict(list)
        for lic in ws.licenses.values():
            if lic.holder_id:
                u = ws.users.get(lic.holder_id)
                if u and u.active:
                    by_user[lic.holder_id].append(lic.license_id)
        assert "usr_000001" not in by_user


# ─────────────────────────────────────────────────────────────────────────────
# Additional correctness checks
# ─────────────────────────────────────────────────────────────────────────────


class TestFoldCorrectness:
    def test_contract_renewal_updates_term(self) -> None:
        led = _minimal_ledger()
        T = date(2024, 1, 20)
        ws = fold_events(led.events_through(T))
        # ctr_0001 was initially 12 months, renewed to 24 months on day 10
        assert ws.contracts["ctr_0001"].term_months == 24
        assert ws.contracts["ctr_0001"].last_renewed_at == date(2024, 1, 11)

    def test_contract_not_renewed_unchanged(self) -> None:
        led = _minimal_ledger()
        T = date(2024, 1, 20)
        ws = fold_events(led.events_through(T))
        # ctr_0000 was never renewed
        assert ws.contracts["ctr_0000"].term_months == 12
        assert ws.contracts["ctr_0000"].last_renewed_at is None

    def test_events_through_filters_by_date(self) -> None:
        led = _minimal_ledger()
        events_d0 = led.events_through(date(2024, 1, 1))
        # Only day-0 events (products + contracts + users + licenses)
        assert all(e.at == date(2024, 1, 1) for e in events_d0)

    def test_no_events_before_start(self) -> None:
        led = _minimal_ledger()
        events = led.events_through(date(2023, 12, 31))
        assert events == []

    def test_docs_contain_vendor_name(self, tmp_path: Path) -> None:
        led = _minimal_ledger()
        proj = Projector()
        proj.project_docs(led, date(2024, 1, 20), tmp_path)
        doc = (tmp_path / "ctr_0000.md").read_text()
        assert "Acme Corp" in doc  # vendor_idx=0 → "Acme Corp"

    def test_docs_expired_contract_marked(self, tmp_path: Path) -> None:
        """A contract whose term expired before T should say 'expired'."""
        led = _minimal_ledger()
        # Project far into the future — 12-month term from 2024-01-01 expires ~2025-01-01
        T_far = date(2025, 6, 1)
        proj = Projector()
        proj.project_docs(led, T_far, tmp_path)
        # ctr_0000 has 12-month term, never renewed — must be expired
        doc = (tmp_path / "ctr_0000.md").read_text()
        assert "expired" in doc
