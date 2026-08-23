"""Pure event-fold: canonical derivation of WorldState from a ledger event slice.

This module is intentionally free of I/O, randomness, and store dependencies.
Both the fabric projector and the task resolver import from here; fold logic
lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from churnbench.ledger.ledger import EventKind, LedgerEvent


@dataclass
class User:
    user_id: str
    cost_center_id: str
    hired_at: date
    active: bool = True


@dataclass
class License:
    license_id: str
    product_id: str
    seats: int
    unit_price_usd: float
    purchased_at: date
    holder_id: str | None = None


@dataclass
class Contract:
    contract_id: str
    vendor_idx: int
    term_months: int
    signed_at: date
    last_renewed_at: date | None = None


@dataclass
class Product:
    product_id: str
    current_price: float
    first_seen_at: date


@dataclass
class ConsumptionRow:
    user_id: str
    product_id: str
    event_date: date
    session_minutes: int
    api_calls: int


@dataclass
class WorldState:
    """Accumulated state after folding events[0..T]."""

    users: dict[str, User] = field(default_factory=dict)
    licenses: dict[str, License] = field(default_factory=dict)
    contracts: dict[str, Contract] = field(default_factory=dict)
    products: dict[str, Product] = field(default_factory=dict)
    consumption: list[ConsumptionRow] = field(default_factory=list)


def fold_events(events: list[LedgerEvent]) -> WorldState:
    """Fold a seq-ordered event list into a WorldState snapshot."""
    ws = WorldState()

    for e in events:
        k = e.kind

        if k == EventKind.USER_HIRED:
            ws.users[e.entity_id] = User(
                user_id=e.entity_id,
                cost_center_id=e.payload["cost_center_id"],
                hired_at=e.at,
            )

        elif k == EventKind.USER_OFFBOARDED:
            if e.entity_id in ws.users:
                ws.users[e.entity_id].active = False

        elif k == EventKind.USER_MOVED_COST_CENTER:
            if e.entity_id in ws.users:
                ws.users[e.entity_id].cost_center_id = e.payload["cost_center_id"]

        elif k == EventKind.LICENSE_PURCHASED:
            ws.licenses[e.entity_id] = License(
                license_id=e.entity_id,
                product_id=e.payload["product_id"],
                seats=e.payload["seats"],
                unit_price_usd=float(e.payload["unit_price_usd"]),
                purchased_at=e.at,
            )

        elif k == EventKind.LICENSE_ASSIGNED:
            if e.entity_id in ws.licenses:
                ws.licenses[e.entity_id].holder_id = e.payload["to"]

        elif k == EventKind.LICENSE_UNASSIGNED:
            if e.entity_id in ws.licenses:
                ws.licenses[e.entity_id].holder_id = None

        elif k == EventKind.LICENSE_REASSIGNED:
            if e.entity_id in ws.licenses:
                ws.licenses[e.entity_id].holder_id = e.payload["to"]

        elif k == EventKind.CONTRACT_SIGNED:
            ws.contracts[e.entity_id] = Contract(
                contract_id=e.entity_id,
                vendor_idx=e.payload["vendor_idx"],
                term_months=e.payload["term_months"],
                signed_at=e.at,
            )

        elif k == EventKind.CONTRACT_RENEWED:
            if e.entity_id in ws.contracts:
                ws.contracts[e.entity_id].term_months = e.payload["term_months"]
                ws.contracts[e.entity_id].last_renewed_at = e.at

        elif k == EventKind.PRICE_CHANGED:
            pid = e.entity_id
            if pid not in ws.products:
                ws.products[pid] = Product(
                    product_id=pid,
                    current_price=float(e.payload["unit_price_usd"]),
                    first_seen_at=e.at,
                )
            else:
                ws.products[pid].current_price = float(e.payload["unit_price_usd"])

        elif k == EventKind.CONSUMPTION_LOGGED:
            ws.consumption.append(
                ConsumptionRow(
                    user_id=e.entity_id,
                    product_id=e.payload["product_id"],
                    event_date=e.at,
                    session_minutes=e.payload["session_minutes"],
                    api_calls=e.payload["api_calls"],
                )
            )

    return ws
