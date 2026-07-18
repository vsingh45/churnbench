"""Agentic ETL pipelines — implements Principle 3 (ETL for Agentic Systems).

The staged store is a SQLite database (file-based per run, or in-memory for tests).
`setup()` pulls from origin at T_prime and stamps last_refresh on each entity class.
`refresh_due()` returns the classes whose TTL has lapsed at evaluation time T.

Why SQLite for staging: zero-dependency, embeddable, deterministic query output,
and SQL-queryable — which keeps the router's templated-SQL path identical to the
live-Postgres path in structure (only the table names differ).
"""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import text

from churnbench.arms.grounding.semantic_model import EntityClass

# ── Staged-store DDL ──────────────────────────────────────────────────────────

_STAGED_DDL: list[str] = [
    """CREATE TABLE IF NOT EXISTS staged_users (
        user_id TEXT PRIMARY KEY,
        cost_center_id TEXT,
        active INTEGER,
        hired_at TEXT,
        staged_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS staged_assignments (
        assignment_id TEXT PRIMARY KEY,
        license_id TEXT,
        user_id TEXT,
        product_id TEXT,
        staged_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS staged_license_purchases (
        purchase_id TEXT PRIMARY KEY,
        product_id TEXT,
        cost_center_id TEXT,
        seats INTEGER,
        unit_price_usd REAL,
        valid_from TEXT,
        valid_until TEXT,
        staged_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS staged_cost_centers (
        cost_center_id TEXT PRIMARY KEY,
        cost_center TEXT,
        business_unit TEXT,
        staged_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS staged_vendors (
        vendor_id TEXT PRIMARY KEY,
        vendor_name TEXT,
        vendor_tier TEXT,
        staged_at TEXT
    )""",
]


def create_staged_schema(sqlite_engine: Any) -> None:
    """Create all staging tables in the SQLite store (idempotent)."""
    with sqlite_engine.connect() as conn:
        for stmt in _STAGED_DDL:
            conn.execute(text(stmt))
        conn.commit()


# ── Per-entity ETL pipelines ──────────────────────────────────────────────────


def _refresh_user_status(conn_staged: Any, mongo_db: Any, ts: str) -> None:
    conn_staged.execute(text("DELETE FROM staged_users"))
    users = list(mongo_db["users"].find({}, {"_id": 0}))
    for u in users:
        conn_staged.execute(
            text(
                "INSERT OR REPLACE INTO staged_users "
                "(user_id, cost_center_id, active, hired_at, staged_at) "
                "VALUES (:uid, :cc, :active, :hired, :ts)"
            ),
            {
                "uid": str(u.get("user_id", "")),
                "cc": str(u.get("cost_center_id", "")),
                "active": 1 if u.get("active", True) else 0,
                "hired": str(u.get("hired_at", "")),
                "ts": ts,
            },
        )


def _refresh_assignments(conn_staged: Any, mongo_db: Any, ts: str) -> None:
    conn_staged.execute(text("DELETE FROM staged_assignments"))
    rows = list(mongo_db["assignments"].find({}, {"_id": 0}))
    for r in rows:
        conn_staged.execute(
            text(
                "INSERT OR REPLACE INTO staged_assignments "
                "(assignment_id, license_id, user_id, product_id, staged_at) "
                "VALUES (:id, :lic, :uid, :pid, :ts)"
            ),
            {
                "id": f"{r.get('license_id', '')}__{r.get('user_id', '')}",
                "lic": str(r.get("license_id", "")),
                "uid": str(r.get("user_id", "")),
                "pid": str(r.get("product_id", "")),
                "ts": ts,
            },
        )


def _refresh_prices(conn_staged: Any, pg_engine: Any, ts: str) -> None:
    with pg_engine.connect() as pg:
        rows = pg.execute(
            text(
                "SELECT purchase_id, product_id, cost_center_id, seats, "
                "unit_price_usd, valid_from, valid_until "
                "FROM sam.fact_license_purchase"
            )
        ).mappings().all()
    conn_staged.execute(text("DELETE FROM staged_license_purchases"))
    for r in rows:
        conn_staged.execute(
            text(
                "INSERT OR REPLACE INTO staged_license_purchases "
                "(purchase_id, product_id, cost_center_id, seats, "
                "unit_price_usd, valid_from, valid_until, staged_at) "
                "VALUES (:purchase_id, :product_id, :cost_center_id, :seats, "
                ":unit_price_usd, :valid_from, :valid_until, :ts)"
            ),
            dict(r) | {"ts": ts},
        )


def _refresh_cost_center_membership(
    conn_staged: Any, mongo_db: Any, pg_engine: Any | None, ts: str
) -> None:
    """Derives cost-centre list from MongoDB users; augments names from Postgres dim."""
    cc_ids: list[str] = sorted(
        {str(u.get("cost_center_id", "")) for u in mongo_db["users"].find({}, {"_id": 0})}
    )
    cc_meta: dict[str, dict[str, str]] = {}
    if pg_engine is not None:
        try:
            with pg_engine.connect() as pg:
                for r in pg.execute(
                    text("SELECT cost_center_id, cost_center, business_unit FROM sam.dim_cost_center")
                ).mappings():
                    cc_meta[str(r["cost_center_id"])] = {
                        "cost_center": str(r["cost_center"]),
                        "business_unit": str(r["business_unit"]),
                    }
        except Exception:
            pass

    conn_staged.execute(text("DELETE FROM staged_cost_centers"))
    for cc_id in cc_ids:
        extra = cc_meta.get(cc_id, {})
        conn_staged.execute(
            text(
                "INSERT OR REPLACE INTO staged_cost_centers "
                "(cost_center_id, cost_center, business_unit, staged_at) "
                "VALUES (:ccid, :cc, :bu, :ts)"
            ),
            {
                "ccid": cc_id,
                "cc": extra.get("cost_center", cc_id),
                "bu": extra.get("business_unit", ""),
                "ts": ts,
            },
        )


def _refresh_vendor_dims(conn_staged: Any, pg_engine: Any, ts: str) -> None:
    with pg_engine.connect() as pg:
        rows = pg.execute(
            text("SELECT vendor_id, vendor_name, vendor_tier FROM sam.dim_vendor")
        ).mappings().all()
    conn_staged.execute(text("DELETE FROM staged_vendors"))
    for r in rows:
        conn_staged.execute(
            text(
                "INSERT OR REPLACE INTO staged_vendors "
                "(vendor_id, vendor_name, vendor_tier, staged_at) "
                "VALUES (:vendor_id, :vendor_name, :vendor_tier, :ts)"
            ),
            dict(r) | {"ts": ts},
        )


# Public ETL entry points ──────────────────────────────────────────────────────


def refresh_entity(
    entity_name: str,
    staged_engine: Any,
    *,
    pg_engine: Any = None,
    mongo_db: Any = None,
    T_prime: date,
) -> None:
    """Pull one entity class from its origin and write to staged SQLite.

    Silently no-ops if the required connection is unavailable (tolerates tests
    that only provide one of pg_engine / mongo_db).
    """
    ts = T_prime.isoformat()
    try:
        with staged_engine.connect() as conn:
            if entity_name == "user_status" and mongo_db is not None:
                _refresh_user_status(conn, mongo_db, ts)
            elif entity_name == "assignments" and mongo_db is not None:
                _refresh_assignments(conn, mongo_db, ts)
            elif entity_name == "prices" and pg_engine is not None:
                _refresh_prices(conn, pg_engine, ts)
            elif entity_name == "cost_center_membership" and mongo_db is not None:
                _refresh_cost_center_membership(conn, mongo_db, pg_engine, ts)
            elif entity_name == "vendor_dims" and pg_engine is not None:
                _refresh_vendor_dims(conn, pg_engine, ts)
            # contract_terms → ChromaDB (handled by arm.py setup)
            # consumption_facts / utilization_current / tickets → live-only, never staged
            conn.commit()
    except Exception:
        pass  # connection error → leave table empty; router will fall through to origin


def setup(
    registry: dict[str, EntityClass],
    staged_engine: Any,
    *,
    pg_engine: Any = None,
    mongo_db: Any = None,
    T_prime: date,
) -> None:
    """Run all ETL pipelines at T_prime and stamp last_refresh in the registry.

    Mutates registry in-place (sets last_refresh on each non-live entity class).
    """
    create_staged_schema(staged_engine)
    for name, ec in registry.items():
        if ec.tier == "live" or (ec.staged_table is None and name != "contract_terms"):
            continue  # live-only — never staged
        if name == "contract_terms":
            # ChromaDB index is built by GroundingArm.setup(), not here.
            # Still stamp last_refresh so refresh_due() tracks it correctly.
            ec.last_refresh = T_prime
            continue
        refresh_entity(name, staged_engine, pg_engine=pg_engine, mongo_db=mongo_db, T_prime=T_prime)
        ec.last_refresh = T_prime


def refresh_due(registry: dict[str, EntityClass], T: date) -> list[EntityClass]:
    """Return entity classes whose TTL has lapsed at evaluation time T.

    Live-tier entities are never included — they are always served from origin.
    Never-refreshed stageable entities are always included (last_refresh=None
    makes is_stale_at() return True).
    """
    return [
        ec for ec in registry.values()
        if ec.tier != "live" and ec.is_stale_at(T)
    ]
