"""Semantic model: declarative registry of entity classes for the grounding layer.

Each entity class maps a domain concept to:
  - origin: which enterprise source owns it
  - staged_table: the SQLite staging table (None → not staged as a SQL table)
  - tier: freshness tier (hot / warm / cold / live)
  - ttl_days: cache time-to-live; None means always-live (never staged)
  - measures: resolver_refs this class can answer

Tier assignment follows §3.3's volatility-based rule:
  hot  (TTL 1 day)   — entities that mutate daily (assignments, user status)
  warm (TTL 7 days)  — entities that mutate weekly (prices, cost-center membership)
  cold (TTL 30 days) — entities that mutate rarely or on-event (contracts, vendor dims)
  live (TTL None)    — inherently live: consumption facts, SaaS utilization, tickets
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

Tier = Literal["hot", "warm", "cold", "live"]


@dataclass
class EntityClass:
    """One unit of the semantic model — a stageable or live-only domain concept."""

    name: str
    origin: str  # "postgres" | "mongo" | "saas" | "docs"
    staged_table: str | None  # SQLite table; None for live-only or docs-index
    tier: Tier
    ttl_days: int | None  # None → never staged / always live
    measures: list[str]  # resolver_ref names answerable from this class
    last_refresh: date | None = None

    def is_stale_at(self, T: date) -> bool:
        """Return True if the cached data is past its TTL at evaluation time T."""
        if self.last_refresh is None:
            return True  # never refreshed → treat as stale
        if self.ttl_days is None:
            return False  # live-only → no TTL concept
        return (T - self.last_refresh).days > self.ttl_days


# ── SAM fabric entity registry ────────────────────────────────────────────────
# DATA, not prompts — the router reads metadata here; the LLM reads vocabulary
# from build_need_resolution_prompt() in arm.py.

ENTITY_REGISTRY: dict[str, EntityClass] = {
    # ── Hot tier: daily volatility ─────────────────────────────────────────────
    "assignments": EntityClass(
        name="assignments",
        origin="mongo",
        staged_table="staged_assignments",
        tier="hot",
        ttl_days=1,
        measures=[
            "assigned_license_count",
            "assignment_count",
            "unassigned_license_count",
            "idle_license_count",       # SO1: by product_id (federated)
            "zero_usage_license_count",  # UT4: by product_id (federated)
            # idle_license_count_cc lives under user_status — shared ownership caused
            # need-resolution to sometimes pick zero_usage_license_count (product filter)
            # for cost-center questions.  Keeping it in only one entity class removes
            # the ambiguity.
            # license_reclaim_count removed: requires unassignment event log, not in staged store.
            # SO5 tasks route to docs_index fallback and score as reasoning_error (intentional).
        ],
    ),
    "user_status": EntityClass(
        name="user_status",
        origin="mongo",
        staged_table="staged_users",
        tier="hot",
        ttl_days=1,
        measures=[
            "active_user_count_cc",
            "offboard_count_cc",
            "idle_license_count_cc",
            "user_count_cc",
        ],
    ),
    # ── Warm tier: weekly volatility ───────────────────────────────────────────
    "prices": EntityClass(
        name="prices",
        origin="postgres",
        staged_table="staged_license_purchases",
        tier="warm",
        ttl_days=7,
        measures=[
            "monthly_spend_cc",
            "unit_price_product",
            "total_annual_spend",
            "seat_count_product_cc",
            "top_spending_cost_center",
            "cost_centers_above_threshold",
        ],
    ),
    "cost_center_membership": EntityClass(
        name="cost_center_membership",
        origin="mongo",
        staged_table="staged_cost_centers",
        tier="warm",
        ttl_days=7,
        measures=[
            "cost_center_name",
            "business_unit",
        ],
    ),
    # ── Cold tier: event-driven / ≤monthly volatility ──────────────────────────
    "contract_terms": EntityClass(
        name="contract_terms",
        origin="docs",
        staged_table=None,  # staged in ChromaDB, not a SQLite table
        tier="cold",
        ttl_days=30,
        measures=[
            "contract_annual_value",
            "contract_vendor_name",
            "contract_renewal_date",
            "contract_duration_years",
        ],
    ),
    "vendor_dims": EntityClass(
        name="vendor_dims",
        origin="postgres",
        staged_table="staged_vendors",
        tier="cold",
        ttl_days=30,
        measures=[
            "vendor_name",
            "vendor_tier",
        ],
    ),
    # ── Live tier: inherently live (§3.1 stageable/live classification) ─────────
    "consumption_facts": EntityClass(
        name="consumption_facts",
        origin="postgres",
        staged_table=None,  # warehouse passthrough — never cached
        tier="live",
        ttl_days=None,
        measures=[
            "total_session_minutes",
            "total_api_calls",
            "distinct_active_users_product",
            "top_product_by_session_minutes",
        ],
    ),
    "utilization_current": EntityClass(
        name="utilization_current",
        origin="saas",
        staged_table=None,
        tier="live",
        ttl_days=None,
        measures=[
            "current_utilization_product",
        ],
    ),
    "tickets": EntityClass(
        name="tickets",
        origin="saas",
        staged_table=None,
        tier="live",
        ttl_days=None,
        measures=[
            "open_ticket_count",
            "ticket_status",
        ],
    ),
}

# Flat lookup: measure name → entity class name (for need-resolution inference)
MEASURE_TO_ENTITY: dict[str, str] = {
    measure: name for name, ec in ENTITY_REGISTRY.items() for measure in ec.measures
}
