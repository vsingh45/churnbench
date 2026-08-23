"""ChurnBench fabric projector — materialises the ledger at a frozen timestamp T.

The projector folds ledger.events_through(T) into each downstream store so that
every fabric layer reflects the world state at exactly T.  Running the same
projection twice at the same T yields bit-identical results (idempotency).

Design invariants:
  1. The ledger is the ONLY source of truth.  Projectors never invent facts
     beyond what the event stream contains; TODO comments mark the few
     synthesis choices required by the Postgres star schema.
  2. Every projection wipes its target before rebuilding, guaranteeing
     reproducibility — no partial updates, no leftover state from a prior run.
  3. No random draws.  Determinism is enforced by processing events in seq
     order (the ledger's natural order after events_through filters).
  4. The three projection targets (Postgres, MongoDB, docs) are independent;
     they may be called in any order or in isolation.

Event taxonomy recap (from ledger.py):
  Users:    USER_HIRED, USER_OFFBOARDED, USER_MOVED_COST_CENTER
  Licenses: LICENSE_PURCHASED, LICENSE_ASSIGNED, LICENSE_UNASSIGNED,
            LICENSE_REASSIGNED
  Contracts: CONTRACT_SIGNED, CONTRACT_RENEWED
  Pricing:  PRICE_CHANGED
  Consumption: CONSUMPTION_LOGGED  (append-only, no state mutation)
"""

from __future__ import annotations

import hashlib
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from jinja2 import Environment as JinjaEnv
from jinja2 import StrictUndefined, select_autoescape
from pymongo.database import Database as MongoDatabase
from sqlalchemy import text
from sqlalchemy.engine import Connection

from churnbench.ledger.fold import (
    User as _User,
    WorldState as _WorldState,
    fold_events as _fold_events,
)
from churnbench.ledger.ledger import EventKind, Ledger, LedgerEvent


# ─────────────────────────────────────────────────────────────────────────────
# Dimension synthesis helpers
#
# The Postgres star schema has dim_vendor, dim_cost_center, dim_product — none
# of which carry their own ledger events.  We synthesise them deterministically
# from the event stream.  Each TODO marks a synthesis decision worth revisiting.
# ─────────────────────────────────────────────────────────────────────────────


def vendor_name(vendor_idx: int) -> str:
    """Synthesise a stable vendor name from the integer index in CONTRACT_SIGNED.

    TODO: replace with a real vendor registry event type if vendor metadata
    (address, tier, etc.) is needed for more realistic grounding tasks.
    """
    names = [
        "Acme Corp",
        "Nexus Software",
        "Orbit Systems",
        "Pinnacle Tech",
        "Quantum Labs",
        "Rapid Solutions",
        "Stellar Dynamics",
        "Titan Networks",
        "Unity Platforms",
        "Vanguard Analytics",
        "Warp Digital",
        "Xenon Cloud",
    ]
    return names[vendor_idx % len(names)]


def vendor_tier(vendor_idx: int) -> str:
    """Assign a tier based on index so distribution is deterministic.

    TODO: could be driven by contract value or seat count once those facts
    are tracked per-vendor in the ledger.
    """
    if vendor_idx % 3 == 0:
        return "strategic"
    if vendor_idx % 3 == 1:
        return "preferred"
    return "tail"


def _product_name(product_id: str) -> str:
    """Synthesise a human-readable product name from its SKU.

    TODO: a PRODUCT_CREATED event type would let us store a real name at
    creation time rather than deriving it here.
    """
    # prd_0001 → "Product 0001"
    suffix = product_id.split("_")[-1] if "_" in product_id else product_id
    return f"Product {suffix}"


def _license_model(product_id: str) -> str:
    """Deterministic license model based on product index parity.

    TODO: introduce explicit product metadata (either in the ledger or a
    separate config) to drive more varied licensing models.
    """
    idx = int(product_id.split("_")[-1]) if "_" in product_id else 0
    models = ["user", "device", "concurrent", "consumption"]
    return models[idx % 4]


def _cc_business_unit(cost_center_id: str) -> str:
    """Synthesise a business unit for each cost center.

    TODO: a COST_CENTER_CREATED event would be the right carrier for this.
    """
    idx = int(cost_center_id.split("_")[-1]) if "_" in cost_center_id else 0
    units = ["Engineering", "Sales", "Marketing", "Finance", "HR", "Operations"]
    return units[idx % len(units)]


def _fiscal_quarter(d: date) -> str:
    return f"FY{d.year}Q{(d.month - 1) // 3 + 1}"


# ─────────────────────────────────────────────────────────────────────────────
# Projector
# ─────────────────────────────────────────────────────────────────────────────


class Projector:
    """Materialises ground-truth ledger state at a chosen timestamp T.

    Each method is idempotent: calling it twice with the same arguments leaves
    the store in the same state as calling it once.  The wipe-then-rebuild
    strategy is deliberate — partial updates would risk leaving stale rows if
    an event retroactively removes an entity.
    """

    # ── Postgres ──────────────────────────────────────────────────────────

    def project_postgres(self, ledger: Ledger, T: date) -> dict[str, int]:
        """Wipe and rebuild the sam.* tables to reflect the world at T.

        Returns row counts per table for logging / testing.
        """
        from churnbench.fabric.connections import pg_engine

        events = ledger.events_through(T)
        ws = _fold_events(events)

        engine = pg_engine()
        with engine.begin() as conn:
            self._pg_wipe(conn)
            vendors = self._pg_insert_vendors(conn, ws)
            cost_centers = self._pg_insert_cost_centers(conn, ws)
            products = self._pg_insert_products(conn, ws, vendors)
            n_purchases = self._pg_insert_license_purchases(conn, ws, products, cost_centers, T)
            n_consumption = self._pg_insert_consumption(conn, ws, products)
            n_dates = self._pg_insert_dates(conn, T)

        return {
            "dim_vendor": len(vendors),
            "dim_cost_center": len(cost_centers),
            "dim_product": len(products),
            "fact_license_purchase": n_purchases,
            "fact_consumption_event": n_consumption,
            "dim_date": n_dates,
        }

    def _pg_wipe(self, conn: Connection) -> None:
        # Truncate in FK-safe order (facts before dims)
        conn.execute(text("TRUNCATE sam.fact_consumption_event RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE sam.fact_license_purchase RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE sam.dim_product RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE sam.dim_cost_center RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE sam.dim_vendor RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE sam.dim_date CASCADE"))

    def _pg_insert_vendors(self, conn: Connection, ws: _WorldState) -> dict[int, int]:
        """Insert one row per distinct vendor_idx seen in CONTRACT_SIGNED events.

        Returns {vendor_idx -> serial vendor_id}.
        """
        seen: dict[int, int] = {}
        for ctr in ws.contracts.values():
            idx = ctr.vendor_idx
            if idx in seen:
                continue
            result = conn.execute(
                text(
                    "INSERT INTO sam.dim_vendor (vendor_name, vendor_tier) "
                    "VALUES (:name, :tier) "
                    "ON CONFLICT (vendor_name) DO UPDATE SET vendor_tier = EXCLUDED.vendor_tier "
                    "RETURNING vendor_id"
                ),
                {"name": vendor_name(idx), "tier": vendor_tier(idx)},
            )
            row = result.fetchone()
            seen[idx] = row[0]  # type: ignore[index]
        return seen

    def _pg_insert_cost_centers(self, conn: Connection, ws: _WorldState) -> dict[str, int]:
        """Insert one row per cost_center_id seen in user events.

        Returns {cost_center_id -> serial cost_center_id (PK)}.
        """
        # Collect every cost_center_id referenced by any user (current or former).
        cc_ids: set[str] = set()
        for u in ws.users.values():
            cc_ids.add(u.cost_center_id)

        mapping: dict[str, int] = {}
        for cc_id in sorted(cc_ids):  # sorted → deterministic insert order
            result = conn.execute(
                text(
                    "INSERT INTO sam.dim_cost_center (cost_center, business_unit) "
                    "VALUES (:cc, :bu) "
                    "ON CONFLICT (cost_center) DO UPDATE SET business_unit = EXCLUDED.business_unit "
                    "RETURNING cost_center_id"
                ),
                {"cc": cc_id, "bu": _cc_business_unit(cc_id)},
            )
            row = result.fetchone()
            mapping[cc_id] = row[0]  # type: ignore[index]
        return mapping

    def _pg_insert_products(
        self, conn: Connection, ws: _WorldState, vendors: dict[int, int]
    ) -> dict[str, int]:
        """Insert one row per product seen in PRICE_CHANGED events.

        vendor_id is assigned by round-robining over known vendors.
        Returns {product_id -> serial product_id (PK)}.

        TODO: assign vendor per product via a PRODUCT_ASSIGNED_VENDOR event
        rather than round-robin so vendor attribution is meaningful.
        """
        vendor_pks = sorted(vendors.values())
        mapping: dict[str, int] = {}

        for i, (pid, prod) in enumerate(sorted(ws.products.items())):
            vendor_pk = vendor_pks[i % len(vendor_pks)] if vendor_pks else None
            result = conn.execute(
                text(
                    "INSERT INTO sam.dim_product "
                    "(product_sku, product_name, vendor_id, license_model) "
                    "VALUES (:sku, :name, :vid, :model) "
                    "ON CONFLICT (product_sku) DO UPDATE "
                    "SET product_name = EXCLUDED.product_name, "
                    "    vendor_id    = EXCLUDED.vendor_id, "
                    "    license_model = EXCLUDED.license_model "
                    "RETURNING product_id"
                ),
                {
                    "sku": pid,
                    "name": _product_name(pid),
                    "vid": vendor_pk,
                    "model": _license_model(pid),
                },
            )
            row = result.fetchone()
            mapping[pid] = row[0]  # type: ignore[index]
        return mapping

    def _pg_insert_license_purchases(
        self,
        conn: Connection,
        ws: _WorldState,
        products: dict[str, int],
        cost_centers: dict[str, int],
        T: date,
    ) -> int:
        """Populate fact_license_purchase — one row per license in the world state.

        cost_center_id is taken from the current holder's cost center (if assigned),
        otherwise left as the first cost center (a synthetic default).

        TODO: track the cost center at purchase time in the LICENSE_PURCHASED
        payload so the fact table reflects who paid, not who holds it today.
        """
        default_cc = min(cost_centers.values()) if cost_centers else None

        rows = []
        for lic in ws.licenses.values():
            product_pk = products.get(lic.product_id)
            if product_pk is None:
                continue  # product not yet seen at T; skip
            holder = ws.users.get(lic.holder_id) if lic.holder_id else None
            cc_pk = cost_centers.get(holder.cost_center_id, default_cc) if holder else default_cc
            valid_from = lic.purchased_at
            valid_until = valid_from + timedelta(days=365)
            rows.append(
                {
                    "product_id": product_pk,
                    "cost_center_id": cc_pk,
                    "purchase_date": valid_from,
                    "seats": lic.seats,
                    "unit_price_usd": lic.unit_price_usd,
                    "contract_id": None,  # TODO: link via PRODUCT→CONTRACT when events carry it
                    "valid_from": valid_from,
                    "valid_until": valid_until,
                }
            )

        if rows:
            conn.execute(
                text(
                    "INSERT INTO sam.fact_license_purchase "
                    "(product_id, cost_center_id, purchase_date, seats, "
                    " unit_price_usd, contract_id, valid_from, valid_until) "
                    "VALUES (:product_id, :cost_center_id, :purchase_date, :seats, "
                    "        :unit_price_usd, :contract_id, :valid_from, :valid_until)"
                ),
                rows,
            )
        return len(rows)

    def _pg_insert_consumption(
        self, conn: Connection, ws: _WorldState, products: dict[str, int]
    ) -> int:
        rows = []
        for c in ws.consumption:
            product_pk = products.get(c.product_id)
            if product_pk is None:
                continue
            rows.append(
                {
                    "product_id": product_pk,
                    "user_ext_id": c.user_id,
                    "event_date": c.event_date,
                    "session_minutes": c.session_minutes,
                    "api_calls": c.api_calls,
                }
            )

        if rows:
            conn.execute(
                text(
                    "INSERT INTO sam.fact_consumption_event "
                    "(product_id, user_ext_id, event_date, session_minutes, api_calls) "
                    "VALUES (:product_id, :user_ext_id, :event_date, "
                    "        :session_minutes, :api_calls)"
                ),
                rows,
            )
        return len(rows)

    def _pg_insert_dates(self, conn: Connection, T: date) -> int:
        """Populate dim_date for every day in the timeline up to T."""
        # Use the ledger's start date as anchor (always 2024-01-01 in default config)
        start = date(2024, 1, 1)
        dates = []
        cur = start
        while cur <= T:
            dates.append({"date_id": cur, "fq": _fiscal_quarter(cur), "fy": cur.year})
            cur += timedelta(days=1)
        if dates:
            conn.execute(
                text(
                    "INSERT INTO sam.dim_date (date_id, fiscal_quarter, fiscal_year) "
                    "VALUES (:date_id, :fq, :fy) "
                    "ON CONFLICT (date_id) DO NOTHING"
                ),
                dates,
            )
        return len(dates)

    # ── MongoDB ───────────────────────────────────────────────────────────

    def project_mongo(
        self,
        ledger: Ledger,
        T: date,
        db: MongoDatabase | None = None,  # type: ignore[type-arg]
    ) -> dict[str, int]:
        """Wipe and rebuild all sam_ops collections to reflect the world at T.

        Returns document counts per collection.

        Collections rebuilt:
          users              — active users at T (USER_OFFBOARDED removes)
          active_licenses    — licenses with a current holder
          assignments        — (license_id, user_id) pairs; reassignments mutate
          entitlements       — per-user view of what they hold
          tickets            — synthetic lifecycle tickets derived from offboard events
          utilization_current — per-product aggregate of consumption through T
        """
        from churnbench.fabric.connections import mongo_db as get_db

        mdb = db if db is not None else get_db()

        events = ledger.events_through(T)
        ws = _fold_events(events)

        # Wipe all collections
        for coll in (
            "users",
            "active_licenses",
            "assignments",
            "entitlements",
            "tickets",
            "utilization_current",
        ):
            mdb[coll].delete_many({})

        counts: dict[str, int] = {}
        counts["users"] = self._mongo_users(mdb, ws)
        counts["active_licenses"] = self._mongo_active_licenses(mdb, ws)
        counts["assignments"] = self._mongo_assignments(mdb, ws)
        counts["entitlements"] = self._mongo_entitlements(mdb, ws)
        counts["tickets"] = self._mongo_tickets(mdb, ws, events)
        counts["utilization_current"] = self._mongo_utilization(mdb, ws)
        return counts

    def _mongo_users(self, db: Any, ws: _WorldState) -> int:
        active = [
            {
                "user_ext_id": u.user_id,
                "cost_center_id": u.cost_center_id,
                "hired_at": u.hired_at.isoformat(),
                "active": True,
            }
            for u in ws.users.values()
            if u.active
        ]
        if active:
            db["users"].insert_many(active)
        return len(active)

    def _mongo_active_licenses(self, db: Any, ws: _WorldState) -> int:
        docs = [
            {
                "license_id": lic.license_id,
                "product_id": lic.product_id,
                "holder_id": lic.holder_id,
                "seats": lic.seats,
                "unit_price_usd": lic.unit_price_usd,
                "purchased_at": lic.purchased_at.isoformat(),
                "assigned": lic.holder_id is not None,
            }
            for lic in ws.licenses.values()
            if lic.holder_id is not None
        ]
        if docs:
            db["active_licenses"].insert_many(docs)
        return len(docs)

    def _mongo_assignments(self, db: Any, ws: _WorldState) -> int:
        docs = [
            {
                "license_id": lic.license_id,
                "user_ext_id": lic.holder_id,
                "product_id": lic.product_id,
            }
            for lic in ws.licenses.values()
            if lic.holder_id is not None
        ]
        if docs:
            db["assignments"].insert_many(docs)
        return len(docs)

    def _mongo_entitlements(self, db: Any, ws: _WorldState) -> int:
        """Per-user entitlements: group active assignments by user."""
        from collections import defaultdict

        by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for lic in ws.licenses.values():
            if (
                lic.holder_id
                and ws.users.get(lic.holder_id, _User("", "", date.today(), False)).active
            ):
                by_user[lic.holder_id].append(
                    {
                        "license_id": lic.license_id,
                        "product_id": lic.product_id,
                        "seats": lic.seats,
                    }
                )
        docs = [{"user_ext_id": uid, "licenses": lics} for uid, lics in by_user.items()]
        if docs:
            db["entitlements"].insert_many(docs)
        return len(docs)

    def _mongo_tickets(self, db: Any, ws: _WorldState, events: list[LedgerEvent]) -> int:
        """Synthesise one 'closed' offboard ticket per USER_OFFBOARDED event.

        TODO: a TICKET_CREATED event type would let us emit richer SaaS-style
        ticket lifecycles (open → in-progress → resolved) rather than deriving
        closed tickets from offboard events only.
        """
        docs = []
        for e in events:
            if e.kind == EventKind.USER_OFFBOARDED:
                ticket_id = f"tkt_{e.seq:08d}"
                docs.append(
                    {
                        "ticket_id": ticket_id,
                        "user_ext_id": e.entity_id,
                        "type": "offboard",
                        "status": "closed",
                        "created_at": e.at.isoformat(),
                        "product_sku": None,  # offboard tickets aren't product-specific
                    }
                )
        if docs:
            db["tickets"].insert_many(docs)
        return len(docs)

    def _mongo_utilization(self, db: Any, ws: _WorldState) -> int:
        """Per-product utilization aggregate (total sessions + API calls)."""
        from collections import defaultdict

        agg: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"session_minutes": 0, "api_calls": 0, "event_count": 0}
        )
        for c in ws.consumption:
            agg[c.product_id]["session_minutes"] += c.session_minutes
            agg[c.product_id]["api_calls"] += c.api_calls
            agg[c.product_id]["event_count"] += 1

        docs = [
            {
                "product_sku": pid,
                "total_session_minutes": v["session_minutes"],
                "total_api_calls": v["api_calls"],
                "event_count": v["event_count"],
            }
            for pid, v in agg.items()
        ]
        if docs:
            db["utilization_current"].insert_many(docs)
        return len(docs)

    # ── Docs ─────────────────────────────────────────────────────────────

    def project_docs(self, ledger: Ledger, T: date, out_dir: Path) -> int:
        """Render one synthetic contract markdown per active contract at T.

        Files are written to out_dir/<contract_id>.md.  Existing files are
        overwritten (idempotent).  The directory is created if it does not exist.

        Returns the number of contract documents rendered.
        """
        events = ledger.events_through(T)
        ws = _fold_events(events)
        out_dir.mkdir(parents=True, exist_ok=True)

        env = self._jinja_env()
        tmpl = env.from_string(_CONTRACT_TEMPLATE)

        rendered = 0
        for ctr in sorted(ws.contracts.values(), key=lambda c: c.contract_id):
            effective_date = ctr.last_renewed_at or ctr.signed_at
            expiry_date = effective_date + timedelta(days=ctr.term_months * 30)
            # Only render contracts whose term hasn't expired before T
            # (a renewed contract resets the clock)
            if expiry_date < T:
                # Show as expired but still render; an agent should detect drift
                status = "expired"
            else:
                status = "active"

            vendor_idx = ctr.vendor_idx
            ctx: dict[str, Any] = {
                "contract_id": ctr.contract_id,
                "vendor_name": vendor_name(vendor_idx),
                "vendor_tier": vendor_tier(vendor_idx),
                "term_months": ctr.term_months,
                "signed_at": ctr.signed_at.isoformat(),
                "effective_date": effective_date.isoformat(),
                "expiry_date": expiry_date.isoformat(),
                "status": status,
                "as_of": T.isoformat(),
                # Stable "contract value" derived from vendor index so it's
                # deterministic without a random draw.
                # TODO: carry seat count + price in CONTRACT_SIGNED payload so
                # contract value is ledger-derived, not synthesised here.
                "annual_value_usd": stable_contract_value(ctr.contract_id),
            }
            doc_path = out_dir / f"{ctr.contract_id}.md"
            doc_path.write_text(tmpl.render(**ctx))
            rendered += 1

        return rendered

    def _jinja_env(self) -> JinjaEnv:
        return JinjaEnv(
            undefined=StrictUndefined,
            autoescape=select_autoescape([]),
            trim_blocks=True,
            lstrip_blocks=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Contract doc template
# ─────────────────────────────────────────────────────────────────────────────

_CONTRACT_TEMPLATE = """\
# Software License Agreement — {{ contract_id }}

**Vendor:** {{ vendor_name }} (tier: {{ vendor_tier }})
**Status:** {{ status }}
**As of:** {{ as_of }}

## Term

| Field | Value |
|-------|-------|
| Signed | {{ signed_at }} |
| Effective | {{ effective_date }} |
| Expiry | {{ expiry_date }} |
| Duration | {{ term_months }} months |

## Financials

**Estimated Annual Value:** ${{ "%.2f"|format(annual_value_usd) }}

## Terms and Conditions

This Software License Agreement ("Agreement") is entered into between the Customer
and {{ vendor_name }} as of {{ signed_at }}.

1. **License Grant.** {{ vendor_name }} grants Customer a non-exclusive, non-transferable
   license to use the licensed software for the term specified above.

2. **Renewal.** This agreement {% if status == "active" %}is currently active{% else %}has expired{% endif %}
   and {% if status == "active" %}expires on {{ expiry_date }}{% else %}was last effective until {{ expiry_date }}{% endif %}.

3. **Support.** Standard support is included for the duration of the active term.

4. **Confidentiality.** Both parties agree to maintain confidentiality of proprietary
   information disclosed under this Agreement.

5. **Governing Law.** This Agreement shall be governed by applicable commercial law.

---
*Generated by ChurnBench projector — frozen at {{ as_of }}.*
"""


def stable_contract_value(contract_id: str) -> float:
    """Derive a stable contract value from the contract ID via a hash.

    This is purely synthetic — no real value data exists in the ledger.
    TODO: add annual_value_usd to CONTRACT_SIGNED payload so this can be
    driven by the event stream rather than synthesised.
    """
    h = int(hashlib.sha1(contract_id.encode()).hexdigest(), 16)
    # Map to $10k–$500k range
    return round(10_000 + (h % 490_000) + (h % 100) * 0.01, 2)
