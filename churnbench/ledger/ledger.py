"""ChurnBench ledger — the ground-truth record.

Every mutation the generator applies to the simulated fabric is written here
as an append-only event with a monotonic timestamp. Task ground truth is
derived exclusively from this ledger, never from live-queried fabric state,
so evaluation is deterministic and reproducible across arms.

Ledger events are the ONLY authoritative source of truth. The Postgres and
Mongo stores are downstream *projections* of the ledger at the current
timeline cursor.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any


class EventKind(str, Enum):
    # Users
    USER_HIRED = "user_hired"
    USER_OFFBOARDED = "user_offboarded"
    USER_MOVED_COST_CENTER = "user_moved_cost_center"

    # Licenses
    LICENSE_PURCHASED = "license_purchased"
    LICENSE_ASSIGNED = "license_assigned"
    LICENSE_UNASSIGNED = "license_unassigned"
    LICENSE_REASSIGNED = "license_reassigned"

    # Contracts / pricing
    CONTRACT_SIGNED = "contract_signed"
    CONTRACT_RENEWED = "contract_renewed"
    PRICE_CHANGED = "price_changed"

    # Consumption (append-only, does not mutate prior state)
    CONSUMPTION_LOGGED = "consumption_logged"


@dataclass(frozen=True)
class LedgerEvent:
    seq: int  # monotonic — the ONLY tie-breaker
    at: date  # timeline day
    kind: EventKind
    entity_type: str  # 'user' | 'license' | 'contract' | 'product'
    entity_id: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["at"] = self.at.isoformat()
        d["kind"] = self.kind.value
        return d


class Ledger:
    """Append-only, in-memory ledger with disk persistence.

    Not thread-safe by design — the generator is single-threaded and
    deterministic given a seed. Concurrency would defeat reproducibility.
    """

    def __init__(self) -> None:
        self._events: list[LedgerEvent] = []
        self._next_seq: int = 0

    # ── Writing ────────────────────────────────────────────────────────────

    def append(
        self,
        at: date,
        kind: EventKind,
        entity_type: str,
        entity_id: str,
        payload: dict[str, Any] | None = None,
    ) -> LedgerEvent:
        event = LedgerEvent(
            seq=self._next_seq,
            at=at,
            kind=kind,
            entity_type=entity_type,
            entity_id=entity_id,
            payload=dict(payload or {}),
        )
        self._events.append(event)
        self._next_seq += 1
        return event

    # ── Reading ────────────────────────────────────────────────────────────

    def events_through(self, T: date) -> list[LedgerEvent]:
        """All events with at <= T, in seq order.

        Frozen-at-T reads use this: reconstruct any projection by folding
        these events. Correctness is defined here — not by whatever is
        currently sitting in Postgres or Mongo.
        """
        return [e for e in self._events if e.at <= T]

    def events_between(self, T0: date, T1: date) -> list[LedgerEvent]:
        return [e for e in self._events if T0 < e.at <= T1]

    def __len__(self) -> int:
        return len(self._events)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for e in self._events:
                f.write(json.dumps(e.to_jsonable()) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Ledger":
        led = cls()
        with path.open() as f:
            for line in f:
                d = json.loads(line)
                event = LedgerEvent(
                    seq=d["seq"],
                    at=date.fromisoformat(d["at"]),
                    kind=EventKind(d["kind"]),
                    entity_type=d["entity_type"],
                    entity_id=d["entity_id"],
                    payload=d["payload"],
                )
                led._events.append(event)
        led._next_seq = (led._events[-1].seq + 1) if led._events else 0
        return led


def utc_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
