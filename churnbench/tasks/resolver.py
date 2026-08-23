"""LedgerResolver: computes ground-truth answers at arbitrary timestamps.

Every public method is callable as resolver.method(**params, T=at) so Task.gold()
can invoke any resolver_ref by name with a single pattern.  T is always a keyword
argument; domain-specific params are positional-style keyword arguments.

The WorldState cache ensures that repeated calls at the same T fold the event
stream only once — important for batch gold computation across ~180 tasks.
"""

from __future__ import annotations

import hashlib
from datetime import date, timedelta

from churnbench.ledger.fold import Contract, User, WorldState, fold_events
from churnbench.ledger.ledger import EventKind, Ledger
from churnbench.fabric.projector import vendor_name, vendor_tier


class LedgerResolver:
    """Ground-truth answer engine backed exclusively by the event ledger."""

    def __init__(self, ledger: Ledger) -> None:
        self._ledger = ledger
        self._ws_cache: dict[date, WorldState] = {}

    def _ws(self, T: date) -> WorldState:
        if T not in self._ws_cache:
            self._ws_cache[T] = fold_events(self._ledger.events_through(T))
        return self._ws_cache[T]

    # ── Base structural methods ──────────────────────────────────────────────

    def active_users(self, T: date) -> dict[str, User]:
        ws = self._ws(T)
        return {uid: u for uid, u in ws.users.items() if u.active}

    def assignments(self, T: date) -> dict[str, str]:
        ws = self._ws(T)
        return {lid: lic.holder_id for lid, lic in ws.licenses.items() if lic.holder_id is not None}

    def offboarded_users(self, T0: date, T1: date) -> list[str]:
        events = self._ledger.events_between(T0, T1)
        return [e.entity_id for e in events if e.kind == EventKind.USER_OFFBOARDED]

    def product_price(self, product_id: str, T: date) -> float:
        ws = self._ws(T)
        prod = ws.products.get(product_id)
        return prod.current_price if prod is not None else 0.0

    def licenses_assigned_to_offboarded(self, T: date) -> list[tuple[str, str]]:
        ws = self._ws(T)
        result: list[tuple[str, str]] = []
        for lid, lic in ws.licenses.items():
            if lic.holder_id is not None:
                user = ws.users.get(lic.holder_id)
                if user is not None and not user.active:
                    result.append((lid, lic.holder_id))
        return result

    def monthly_spend_by_cost_center(self, T: date) -> dict[str, float]:
        """Monthly license spend run-rate (seats × unit_price) by cost center."""
        ws = self._ws(T)
        spend: dict[str, float] = {}
        for lic in ws.licenses.values():
            if lic.holder_id is None:
                continue
            user = ws.users.get(lic.holder_id)
            if user is None or not user.active:
                continue
            cc = user.cost_center_id
            spend[cc] = spend.get(cc, 0.0) + lic.seats * lic.unit_price_usd
        return spend

    def unused_licenses(self, T: date, idle_days: int) -> list[str]:
        """Assigned license IDs whose holder had no consumption in the last idle_days days."""
        ws = self._ws(T)
        cutoff = T - timedelta(days=idle_days)
        recent: set[tuple[str, str]] = set()
        for row in ws.consumption:
            if row.event_date >= cutoff:
                recent.add((row.user_id, row.product_id))
        result: list[str] = []
        for lid, lic in ws.licenses.items():
            if lic.holder_id is not None and (lic.holder_id, lic.product_id) not in recent:
                result.append(lid)
        return result

    def contract_terms(self, contract_id: str, T: date) -> Contract:
        ws = self._ws(T)
        return ws.contracts[contract_id]

    def utilization(self, product_id: str, T: date, window_days: int) -> dict[str, int]:
        """Aggregate session_minutes, api_calls, unique_users for product in window."""
        ws = self._ws(T)
        cutoff = T - timedelta(days=window_days)
        total_sessions = 0
        total_api_calls = 0
        users: set[str] = set()
        for row in ws.consumption:
            if row.product_id == product_id and row.event_date >= cutoff:
                total_sessions += row.session_minutes
                total_api_calls += row.api_calls
                users.add(row.user_id)
        return {
            "session_minutes": total_sessions,
            "api_calls": total_api_calls,
            "unique_users": len(users),
        }

    # ── Scalar answer methods (called via Task.gold / resolver_ref) ──────────

    def spend_by_cost_center(self, cc: str, T: date) -> float:
        return self.monthly_spend_by_cost_center(T).get(cc, 0.0)

    def total_monthly_spend(self, T: date) -> float:
        return sum(self.monthly_spend_by_cost_center(T).values())

    def top_spending_cost_center(self, T: date) -> str:
        spend = self.monthly_spend_by_cost_center(T)
        if not spend:
            return ""
        return max(spend, key=lambda k: spend[k])

    def cost_centers_above_threshold(self, threshold: float, T: date) -> int:
        return sum(1 for v in self.monthly_spend_by_cost_center(T).values() if v > threshold)

    def idle_license_count(self, product_id: str, idle_days: int, T: date) -> int:
        ws = self._ws(T)
        cutoff = T - timedelta(days=idle_days)
        recent_holders: set[str] = set()
        for row in ws.consumption:
            if row.product_id == product_id and row.event_date >= cutoff:
                recent_holders.add(row.user_id)
        count = 0
        for lic in ws.licenses.values():
            if lic.product_id == product_id and lic.holder_id is not None:
                if lic.holder_id not in recent_holders:
                    count += 1
        return count

    def unassigned_license_count(self, T: date) -> int:
        ws = self._ws(T)
        return sum(1 for lic in ws.licenses.values() if lic.holder_id is None)

    def orphan_license_count(self, T: date) -> int:
        return len(self.licenses_assigned_to_offboarded(T))

    def idle_license_cost(self, idle_days: int, T: date) -> float:
        ws = self._ws(T)
        idle_ids = set(self.unused_licenses(T, idle_days))
        total = 0.0
        for lid in idle_ids:
            lic = ws.licenses.get(lid)
            if lic is not None:
                total += lic.seats * lic.unit_price_usd
        return round(total, 2)

    def idle_license_count_cc(self, cc: str, idle_days: int, T: date) -> int:
        """Assigned licenses in cost center cc whose holder had no consumption in idle_days."""
        ws = self._ws(T)
        cutoff = T - timedelta(days=idle_days)
        recent: set[tuple[str, str]] = set()
        for row in ws.consumption:
            if row.event_date >= cutoff:
                recent.add((row.user_id, row.product_id))
        count = 0
        for lic in ws.licenses.values():
            if lic.holder_id is None:
                continue
            user = ws.users.get(lic.holder_id)
            if user is None or not user.active or user.cost_center_id != cc:
                continue
            if (lic.holder_id, lic.product_id) not in recent:
                count += 1
        return count

    def license_reclaim_count(self, days_back: int, T: date) -> int:
        """LICENSE_UNASSIGNED events in the last days_back days.

        Measures clean-up burden: each reclaim is a license freed from a departing
        user that must be reallocated or cancelled.
        """
        T0 = T - timedelta(days=days_back)
        events = self._ledger.events_between(T0, T)
        return sum(1 for e in events if e.kind == EventKind.LICENSE_UNASSIGNED)

    def offboard_hold_count(self, days_back: int, T: date) -> int:
        """Licenses held by users who were offboarded within the last days_back days."""
        T0 = T - timedelta(days=days_back)
        offboarded = set(self.offboarded_users(T0, T))
        ws = self._ws(T)
        return sum(1 for lic in ws.licenses.values() if lic.holder_id in offboarded)

    def offboard_count_window(self, days_back: int, T: date) -> int:
        T0 = T - timedelta(days=days_back)
        return len(self.offboarded_users(T0, T))

    def active_user_count_cc(self, cc: str, T: date) -> int:
        return sum(1 for u in self.active_users(T).values() if u.cost_center_id == cc)

    def reassignment_count(self, days_back: int, T: date) -> int:
        T0 = T - timedelta(days=days_back)
        events = self._ledger.events_between(T0, T)
        return sum(1 for e in events if e.kind == EventKind.LICENSE_REASSIGNED)

    def contract_status(self, contract_id: str, T: date) -> str:
        ws = self._ws(T)
        ctr = ws.contracts.get(contract_id)
        if ctr is None:
            return "not_found"
        effective = ctr.last_renewed_at or ctr.signed_at
        expiry = effective + timedelta(days=ctr.term_months * 30)
        return "active" if expiry >= T else "expired"

    def contract_months_remaining(self, contract_id: str, T: date) -> float:
        ws = self._ws(T)
        ctr = ws.contracts.get(contract_id)
        if ctr is None:
            return 0.0
        effective = ctr.last_renewed_at or ctr.signed_at
        expiry = effective + timedelta(days=ctr.term_months * 30)
        delta = (expiry - T).days
        return round(max(0.0, delta / 30.0), 1)

    def contract_vendor(self, contract_id: str, T: date) -> str:
        ws = self._ws(T)
        ctr = ws.contracts.get(contract_id)
        if ctr is None:
            return ""
        return vendor_name(ctr.vendor_idx)

    def contract_vendor_tier(self, contract_id: str, T: date) -> str:
        ws = self._ws(T)
        ctr = ws.contracts.get(contract_id)
        if ctr is None:
            return ""
        return vendor_tier(ctr.vendor_idx)

    def contract_annual_value(self, contract_id: str, T: date) -> float:
        ws = self._ws(T)
        if contract_id not in ws.contracts:
            return 0.0
        h = int(hashlib.sha1(contract_id.encode()).hexdigest(), 16)
        return round(10_000 + (h % 490_000) + (h % 100) * 0.01, 2)

    def product_api_calls(self, product_id: str, window_days: int, T: date) -> int:
        return self.utilization(product_id, T, window_days)["api_calls"]

    def product_session_minutes(self, product_id: str, window_days: int, T: date) -> int:
        return self.utilization(product_id, T, window_days)["session_minutes"]

    def product_active_users(self, product_id: str, window_days: int, T: date) -> int:
        return self.utilization(product_id, T, window_days)["unique_users"]

    def top_product_by_session_minutes(self, window_days: int, T: date) -> str:
        ws = self._ws(T)
        if not ws.products:
            return ""
        return max(
            ws.products.keys(),
            key=lambda pid: self.product_session_minutes(pid, window_days, T),
        )

    def top_product_by_api_calls(self, window_days: int, T: date) -> str:
        ws = self._ws(T)
        if not ws.products:
            return ""
        return max(
            ws.products.keys(),
            key=lambda pid: self.product_api_calls(pid, window_days, T),
        )

    def zero_usage_license_count(self, product_id: str, window_days: int, T: date) -> int:
        return self.idle_license_count(product_id, window_days, T)

    def license_efficiency(self, product_id: str, window_days: int, T: date) -> float:
        """Ratio of active API users to licensed seats for a product."""
        ws = self._ws(T)
        total_seats = sum(
            lic.seats
            for lic in ws.licenses.values()
            if lic.product_id == product_id and lic.holder_id is not None
        )
        if total_seats == 0:
            return 0.0
        active = self.product_active_users(product_id, window_days, T)
        return round(active / total_seats, 4)
