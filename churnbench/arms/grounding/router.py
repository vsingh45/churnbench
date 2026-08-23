"""Source-aware retrieval router — implements Principle 2 (Cost-Aware Architecture).

Routing is RULE-BASED from semantic model metadata.  No LLM call to route; the
LLM only sees resolved facts, never raw routing decisions.

Route types:
  staged_sql       — SQLite query over staged tables (templated or LLM-generated)
  warehouse_live   — Postgres passthrough for consumption_facts
  origin_live_saas — SaaS API call (inherently live: utilization, tickets)
  origin_live_mongo — MongoDB live call (TTL-lapse fallthrough for hot/warm Mongo entities)
  docs_index       — ChromaDB vector search for contract_terms

Templated SQL avoids an extra LLM call on the common path (§3.1 efficiency claim).
The query_method field ("templated" vs "llm_generated") lets the eval harness report
the templated-vs-LLM split ratio referenced in the paper's cost breakdown table.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from churnbench.arms.grounding.semantic_model import ENTITY_REGISTRY, EntityClass


# ── Federated join helpers ─────────────────────────────────────────────────────
# Pure Python; receive materialized rows from staged SQLite and live Postgres
# and return a scalar.  No LLM call; no I/O.


def _join_idle_by_product(
    staged_rows: list[dict[str, Any]], live_rows: list[dict[str, Any]]
) -> int:
    """Count assignments for a product whose holder had any consumption in the window."""
    active_users = {str(r["user_ext_id"]) for r in live_rows}
    return sum(1 for r in staged_rows if str(r["user_id"]) not in active_users)


def _join_zero_api_by_product(
    staged_rows: list[dict[str, Any]], live_rows: list[dict[str, Any]]
) -> int:
    """Count assignments for a product whose holder had >0 API calls in the window."""
    active_users = {str(r["user_ext_id"]) for r in live_rows}
    return sum(1 for r in staged_rows if str(r["user_id"]) not in active_users)


def _join_idle_by_cc(staged_rows: list[dict[str, Any]], live_rows: list[dict[str, Any]]) -> int:
    """Count assignments in a cost center where (user, product) had no consumption."""
    active_pairs = {(str(r["user_ext_id"]), str(r["product_sku"])) for r in live_rows}
    return sum(
        1 for r in staged_rows if (str(r["user_id"]), str(r["product_id"])) not in active_pairs
    )


# ── Federated query templates (staged SQLite + live Postgres, joined in Python) ─────────────
# Used by route="federated" for measures that require cross-source joins.
# join_fn receives (staged_rows, live_rows) as lists of plain dicts and returns a scalar.


@dataclass
class FederatedTemplate:
    """Two-source query template: staged SQLite + live Postgres, joined in Python.

    staged_params / warehouse_params list the filter-dict keys each SQL uses.
    join_fn is pure Python — no LLM call.
    """

    staged_sql: str
    warehouse_sql: str
    join_fn: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Any]
    staged_params: list[str]
    warehouse_params: list[str]


FEDERATED_TEMPLATES: dict[str, FederatedTemplate] = {
    # SO1: licenses for product P idle for >N days (holder had zero consumption of P in window)
    "idle_license_count": FederatedTemplate(
        staged_sql=(
            "SELECT license_id, user_id "
            "FROM staged_assignments "
            "WHERE product_id = :product_id"
        ),
        warehouse_sql=(
            "SELECT DISTINCT e.user_ext_id "
            "FROM sam.fact_consumption_event e "
            "JOIN sam.dim_product dp ON e.product_id = dp.product_id "
            "WHERE dp.product_sku = :product_id "
            "AND e.event_date >= :cutoff_date"
        ),
        join_fn=_join_idle_by_product,
        staged_params=["product_id"],
        warehouse_params=["product_id", "cutoff_date"],
    ),
    # UT4: licenses for product P where holder had zero API calls in the window
    "zero_usage_license_count": FederatedTemplate(
        staged_sql=(
            "SELECT license_id, user_id "
            "FROM staged_assignments "
            "WHERE product_id = :product_id"
        ),
        warehouse_sql=(
            "SELECT DISTINCT e.user_ext_id "
            "FROM sam.fact_consumption_event e "
            "JOIN sam.dim_product dp ON e.product_id = dp.product_id "
            "WHERE dp.product_sku = :product_id "
            "AND e.event_date >= :cutoff_date "
            "AND e.api_calls > 0"
        ),
        join_fn=_join_zero_api_by_product,
        staged_params=["product_id"],
        warehouse_params=["product_id", "cutoff_date"],
    ),
    # SO3: licenses in cost center C where (user, product) pair had no consumption in window
    "idle_license_count_cc": FederatedTemplate(
        staged_sql=(
            "SELECT a.license_id, a.user_id, a.product_id "
            "FROM staged_assignments a "
            "JOIN staged_users u ON a.user_id = u.user_id "
            "WHERE u.cost_center_id = :cost_center"
        ),
        warehouse_sql=(
            "SELECT DISTINCT e.user_ext_id, dp.product_sku "
            "FROM sam.fact_consumption_event e "
            "JOIN sam.dim_product dp ON e.product_id = dp.product_id "
            "WHERE e.event_date >= :cutoff_date"
        ),
        join_fn=_join_idle_by_cc,
        staged_params=["cost_center"],
        warehouse_params=["cutoff_date"],
    ),
}


# ── Templated staged-SQL queries ──────────────────────────────────────────────
# Keys are measure names; values are (sql_template, required_param_names).
# Only params that appear as :name in the template are listed — unused keys in
# the filters dict are silently ignored.

STAGED_SQL_TEMPLATES: dict[str, tuple[str, list[str]]] = {
    "active_user_count_cc": (
        "SELECT COUNT(*) AS value FROM staged_users "
        "WHERE cost_center_id = :cost_center AND active = 1",
        ["cost_center"],
    ),
    "user_count_cc": (
        "SELECT COUNT(*) AS value FROM staged_users WHERE cost_center_id = :cost_center",
        ["cost_center"],
    ),
    # idle_license_count_cc is handled as a FEDERATED_TEMPLATE (staged × live consumption)
    # rather than staged-only (active=0 is a different concept from zero-consumption-in-window).
    "assignment_count": (
        "SELECT COUNT(*) AS value FROM staged_assignments",
        [],
    ),
    "assigned_license_count": (
        "SELECT COUNT(DISTINCT license_id) AS value FROM staged_assignments",
        [],
    ),
    "monthly_spend_cc": (
        "SELECT COALESCE(SUM(seats * unit_price_usd), 0) AS value "
        "FROM staged_license_purchases WHERE cost_center_id = :cost_center",
        ["cost_center"],
    ),
    "top_spending_cost_center": (
        # Returns the cost_center_id with the highest total monthly spend.
        # No filter params — compares across all cost centers.
        "SELECT cost_center_id AS value "
        "FROM staged_license_purchases "
        "GROUP BY cost_center_id "
        "ORDER BY SUM(seats * unit_price_usd) DESC "
        "LIMIT 1",
        [],
    ),
    "cost_centers_above_threshold": (
        # Returns the count of cost centers whose total spend exceeds :threshold.
        "SELECT COUNT(*) AS value FROM ("
        "SELECT cost_center_id "
        "FROM staged_license_purchases "
        "GROUP BY cost_center_id "
        "HAVING SUM(seats * unit_price_usd) > :threshold"
        ")",
        ["threshold"],
    ),
    "seat_count_product_cc": (
        "SELECT COALESCE(SUM(seats), 0) AS value "
        "FROM staged_license_purchases "
        "WHERE cost_center_id = :cost_center AND product_id = :product_id",
        ["cost_center", "product_id"],
    ),
    "vendor_name": (
        "SELECT vendor_name AS value FROM staged_vendors WHERE vendor_id = :vendor_id",
        ["vendor_id"],
    ),
    "vendor_tier": (
        "SELECT vendor_tier AS value FROM staged_vendors WHERE vendor_id = :vendor_id",
        ["vendor_id"],
    ),
    "cost_center_name": (
        "SELECT cost_center AS value FROM staged_cost_centers "
        "WHERE cost_center_id = :cost_center",
        ["cost_center"],
    ),
    "business_unit": (
        "SELECT business_unit AS value FROM staged_cost_centers "
        "WHERE cost_center_id = :cost_center",
        ["cost_center"],
    ),
    "unit_price_product": (
        # ORDER BY valid_from DESC so LIMIT 1 picks the most recently valid price,
        # not a random row when multiple price rows exist for the same product.
        "SELECT unit_price_usd AS value FROM staged_license_purchases "
        "WHERE product_id = :product_id ORDER BY valid_from DESC LIMIT 1",
        ["product_id"],
    ),
    "unassigned_license_count": (
        "SELECT "
        "(SELECT COALESCE(SUM(seats), 0) FROM staged_license_purchases) "
        "- (SELECT COUNT(*) FROM staged_assignments) AS value",
        [],
    ),
}

# ── Templated warehouse-live SQL (Postgres passthrough) ───────────────────────
# Avoids a separate LLM SQL-generation call for the known consumption measures.

WAREHOUSE_SQL_TEMPLATES: dict[str, tuple[str, list[str]]] = {
    # fact_consumption_event.product_id is an INTEGER FK into dim_product.
    # Tasks pass string SKUs (e.g. 'prd_0003'), so we join through dim_product
    # and filter on product_sku to avoid a psycopg2 InvalidTextRepresentation error.
    # All three consumption measures also require a :cutoff_date window bound
    # (derived from task.params["window_days"] or ["idle_days"] by arm.py).
    "total_session_minutes": (
        "SELECT COALESCE(SUM(e.session_minutes), 0) AS value "
        "FROM sam.fact_consumption_event e "
        "JOIN sam.dim_product dp ON e.product_id = dp.product_id "
        "WHERE dp.product_sku = :product_id "
        "AND e.event_date >= :cutoff_date",
        ["product_id", "cutoff_date"],
    ),
    "total_api_calls": (
        "SELECT COALESCE(SUM(e.api_calls), 0) AS value "
        "FROM sam.fact_consumption_event e "
        "JOIN sam.dim_product dp ON e.product_id = dp.product_id "
        "WHERE dp.product_sku = :product_id "
        "AND e.event_date >= :cutoff_date",
        ["product_id", "cutoff_date"],
    ),
    "distinct_active_users_product": (
        "SELECT COUNT(DISTINCT e.user_ext_id) AS value "
        "FROM sam.fact_consumption_event e "
        "JOIN sam.dim_product dp ON e.product_id = dp.product_id "
        "WHERE dp.product_sku = :product_id "
        "AND e.event_date >= :cutoff_date",
        ["product_id", "cutoff_date"],
    ),
    "top_product_by_session_minutes": (
        # No product filter — aggregates across all products and returns the top SKU.
        # :cutoff_date bounds the window (from task.params["window_days"]).
        "SELECT dp.product_sku AS value "
        "FROM sam.fact_consumption_event e "
        "JOIN sam.dim_product dp ON e.product_id = dp.product_id "
        "WHERE e.event_date >= :cutoff_date "
        "GROUP BY dp.product_sku "
        "ORDER BY SUM(e.session_minutes) DESC "
        "LIMIT 1",
        ["cutoff_date"],
    ),
}

# ── Keyword heuristics (no_semantic_model ablation) ───────────────────────────
# Ordered by specificity — first match wins when the question is ambiguous.

_KEYWORD_RULES: list[tuple[list[str], str]] = [
    (["contract", "renewal", "duration", "annual value", "clause"], "contract_terms"),
    (["vendor", "tier", "vendor name"], "vendor_dims"),
    (["consumption", "session minute", "api call", "usage history"], "consumption_facts"),
    (["utilization", "current util", "real-time"], "utilization_current"),
    (["ticket", "support", "open issue"], "tickets"),
    (["spend", "cost", "monthly", "annual spend", "price", "seat cost"], "prices"),
    (["cost center", "business unit", "department", "cc_"], "cost_center_membership"),
    (["assign", "license", "seat", "reclaim", "idle license"], "assignments"),
    (["user", "active", "offboard", "hire", "headcount"], "user_status"),
]


def keyword_entity_classes(question: str) -> list[str]:
    """Return entity class names inferred from question keywords (no LLM, no registry)."""
    q = question.lower()
    matched: list[str] = []
    for keywords, entity in _KEYWORD_RULES:
        if any(kw in q for kw in keywords):
            if entity not in matched:
                matched.append(entity)
    return matched or ["user_status"]  # safe fallback


# ── Route decision ─────────────────────────────────────────────────────────────


@dataclass
class RouteDecision:
    """Routing decision for one retrieval need — consumed by GroundingArm._execute_one()."""

    route: str  # "staged_sql" | "warehouse_live" | "origin_live_saas" |
    #                        "origin_live_mongo" | "docs_index"
    entity_class: str
    measure: str | None
    sql_template: str | None  # pre-built SQL template (staged or warehouse)
    sql_params: list[str]  # parameter names used in the template
    query_method: str  # "templated" | "llm_generated" | "vector_search" | "api_call"
    last_refresh: date | None  # ec.last_refresh at decision time
    cache_miss_reason: str | None  # "ttl_expired" | "never_refreshed" | None


def decide(
    entity_class_name: str,
    measure: str | None,
    T: date,
    filters: dict[str, Any],  # noqa: ARG001 — reserved for future template param selection
    registry: dict[str, EntityClass] | None = None,
    *,
    no_freshness_tiers: bool = False,
    no_source_routing: bool = False,
) -> RouteDecision:
    """Return a RouteDecision for one (entity_class, measure) need at time T.

    Decision priority (highest wins):
      1. no_source_routing → always docs_index
      2. Unknown entity    → docs_index (graceful fallback)
      3. live tier         → warehouse_live (postgres) or origin_live_saas
      4. contract_terms    → docs_index (always vector search, regardless of TTL)
      5. cacheable + fresh → staged_sql
      6. cacheable + stale → origin_live_mongo / warehouse_live (TTL-lapse fallthrough)
    """
    reg = registry or ENTITY_REGISTRY

    # ── no_source_routing ablation: everything through vector index ───────────
    if no_source_routing:
        return RouteDecision(
            route="docs_index",
            entity_class=entity_class_name,
            measure=measure,
            sql_template=None,
            sql_params=[],
            query_method="vector_search",
            last_refresh=None,
            cache_miss_reason=None,
        )

    ec = reg.get(entity_class_name)
    if ec is None:
        return RouteDecision(
            route="docs_index",
            entity_class=entity_class_name,
            measure=measure,
            sql_template=None,
            sql_params=[],
            query_method="vector_search",
            last_refresh=None,
            cache_miss_reason="unknown_entity",
        )

    # ── Live tier: always goes to origin ─────────────────────────────────────
    if ec.tier == "live":
        if ec.origin == "postgres":
            tmpl, params = WAREHOUSE_SQL_TEMPLATES.get(measure or "", (None, []))
            return RouteDecision(
                route="warehouse_live",
                entity_class=entity_class_name,
                measure=measure,
                sql_template=tmpl,
                sql_params=list(params),
                query_method="templated" if tmpl else "llm_generated",
                last_refresh=None,
                cache_miss_reason=None,
            )
        # SaaS live
        return RouteDecision(
            route="origin_live_saas",
            entity_class=entity_class_name,
            measure=measure,
            sql_template=None,
            sql_params=[],
            query_method="api_call",
            last_refresh=None,
            cache_miss_reason=None,
        )

    # ── contract_terms: always docs_index ────────────────────────────────────
    if entity_class_name == "contract_terms":
        return RouteDecision(
            route="docs_index",
            entity_class=entity_class_name,
            measure=measure,
            sql_template=None,
            sql_params=[],
            query_method="vector_search",
            last_refresh=ec.last_refresh,
            cache_miss_reason=None,
        )

    # ── Cacheable tier: check TTL (unless no_freshness_tiers disables it) ────
    if not no_freshness_tiers and ec.is_stale_at(T):
        miss = "never_refreshed" if ec.last_refresh is None else "ttl_expired"
        if ec.origin == "mongo":
            return RouteDecision(
                route="origin_live_mongo",
                entity_class=entity_class_name,
                measure=measure,
                sql_template=None,
                sql_params=[],
                query_method="api_call",
                last_refresh=ec.last_refresh,
                cache_miss_reason=miss,
            )
        # postgres warm/cold (e.g. vendor_dims) → warehouse passthrough
        return RouteDecision(
            route="warehouse_live",
            entity_class=entity_class_name,
            measure=measure,
            sql_template=None,
            sql_params=[],
            query_method="llm_generated",
            last_refresh=ec.last_refresh,
            cache_miss_reason=miss,
        )

    # ── Serve from staged SQL (fresh cache hit) ───────────────────────────────
    if measure and measure in STAGED_SQL_TEMPLATES:
        tmpl, params = STAGED_SQL_TEMPLATES[measure]
        return RouteDecision(
            route="staged_sql",
            entity_class=entity_class_name,
            measure=measure,
            sql_template=tmpl,
            sql_params=list(params),
            query_method="templated",
            last_refresh=ec.last_refresh,
            cache_miss_reason=None,
        )

    # ── Federated template (staged SQLite + live Postgres join) ──────────────
    if measure and measure in FEDERATED_TEMPLATES:
        return RouteDecision(
            route="federated",
            entity_class=entity_class_name,
            measure=measure,
            sql_template=None,
            sql_params=[],
            query_method="federated_template",
            last_refresh=ec.last_refresh,
            cache_miss_reason=None,
        )

    # No template for this measure → LLM will generate SQL
    return RouteDecision(
        route="staged_sql",
        entity_class=entity_class_name,
        measure=measure,
        sql_template=None,
        sql_params=[],
        query_method="llm_generated",
        last_refresh=ec.last_refresh,
        cache_miss_reason=None,
    )
