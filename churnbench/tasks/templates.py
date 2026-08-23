"""Task templates: 22 parameterised question patterns across 4 intents and 3 tiers.

Each TaskTemplate pairs a question string (with {param} placeholders) with:
  - a resolver_ref: the LedgerResolver method name that computes the gold answer
  - a sampler: draws concrete params from the WorldState; returns None if degenerate

Tier 1: single-source lookup (SQL key-value or simple filter)
Tier 2: multi-step join or aggregation across sources
Tier 3: requires reading contract documents (but gold still computed from ledger)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from churnbench.ledger.fold import WorldState

_Sampler = Callable[[WorldState, random.Random], "dict[str, Any] | None"]


@dataclass
class TaskTemplate:
    template_id: str
    intent: str
    tier: int
    question_template: str
    answer_type: str
    resolver_ref: str
    sampler: _Sampler = field(compare=False, repr=False)

    def sample_params(self, ws: WorldState, rng: random.Random) -> dict[str, Any] | None:
        return self.sampler(ws, rng)


# ── Sampler helpers ──────────────────────────────────────────────────────────


def _active_ccs(ws: WorldState) -> list[str]:
    return sorted({u.cost_center_id for u in ws.users.values() if u.active})


def _products_with_licenses(ws: WorldState) -> list[str]:
    return sorted({lic.product_id for lic in ws.licenses.values() if lic.holder_id})


def _products_with_consumption(ws: WorldState) -> list[str]:
    return sorted({row.product_id for row in ws.consumption})


def _idle_days_choices() -> list[int]:
    return [7, 14, 30, 60]


def _days_back_choices() -> list[int]:
    return [7, 14, 30, 60, 90]


def _window_days_choices() -> list[int]:
    return [7, 14, 30]


# ── spend_visibility samplers ────────────────────────────────────────────────


def _samp_sv1(ws: WorldState, rng: random.Random) -> dict[str, Any] | None:
    ccs = _active_ccs(ws)
    return {"cc": rng.choice(ccs)} if ccs else None


def _samp_sv3(ws: WorldState, rng: random.Random) -> dict[str, Any] | None:
    pids = sorted(ws.products.keys())
    return {"product_id": rng.choice(pids)} if pids else None


def _samp_sv4(ws: WorldState, rng: random.Random) -> dict[str, Any] | None:
    return {} if ws.users else None


def _samp_sv5(ws: WorldState, rng: random.Random) -> dict[str, Any] | None:
    has_spend = any(
        lic.holder_id and ws.users.get(lic.holder_id) is not None and ws.users[lic.holder_id].active
        for lic in ws.licenses.values()
    )
    return {} if has_spend else None


def _samp_sv6(ws: WorldState, rng: random.Random) -> dict[str, Any] | None:
    spend: dict[str, float] = {}
    for lic in ws.licenses.values():
        if lic.holder_id is None:
            continue
        u = ws.users.get(lic.holder_id)
        if u is None or not u.active:
            continue
        cc = u.cost_center_id
        spend[cc] = spend.get(cc, 0.0) + lic.seats * lic.unit_price_usd
    if not spend:
        return None
    values = sorted(spend.values())
    threshold = values[max(0, len(values) // 4)]
    return {"threshold": round(threshold, 2)}


# ── savings_opportunity samplers ─────────────────────────────────────────────


def _samp_so1(ws: WorldState, rng: random.Random) -> dict[str, Any] | None:
    pids = _products_with_licenses(ws)
    if not pids:
        return None
    return {"product_id": rng.choice(pids), "idle_days": rng.choice(_idle_days_choices())}


def _samp_so4(ws: WorldState, rng: random.Random) -> dict[str, Any] | None:
    return {"idle_days": rng.choice(_idle_days_choices())} if ws.licenses else None


def _samp_so3(ws: WorldState, rng: random.Random) -> dict[str, Any] | None:
    ccs_with_licenses = sorted(
        {
            ws.users[lic.holder_id].cost_center_id
            for lic in ws.licenses.values()
            if lic.holder_id
            and ws.users.get(lic.holder_id) is not None
            and ws.users[lic.holder_id].active
        }
    )
    if not ccs_with_licenses:
        return None
    return {
        "cc": rng.choice(ccs_with_licenses),
        "idle_days": rng.choice(_idle_days_choices()),
    }


def _samp_so5(ws: WorldState, rng: random.Random) -> dict[str, Any] | None:
    # Bias toward longer windows to capture real unassignment events
    return {"days_back": rng.choice([30, 60, 90, 120, 180])} if ws.licenses else None


# ── criticality samplers ─────────────────────────────────────────────────────


def _samp_cr1(ws: WorldState, rng: random.Random) -> dict[str, Any] | None:
    return {"days_back": rng.choice(_days_back_choices())} if ws.users else None


def _samp_cr2(ws: WorldState, rng: random.Random) -> dict[str, Any] | None:
    return {"days_back": rng.choice(_days_back_choices())} if ws.licenses else None


def _samp_cr_contract(ws: WorldState, rng: random.Random) -> dict[str, Any] | None:
    cids = sorted(ws.contracts.keys())
    return {"contract_id": rng.choice(cids)} if cids else None


# ── utilization samplers ─────────────────────────────────────────────────────


def _samp_ut_product_consumption(ws: WorldState, rng: random.Random) -> dict[str, Any] | None:
    pids = _products_with_consumption(ws)
    if not pids:
        return None
    return {"product_id": rng.choice(pids), "window_days": rng.choice(_window_days_choices())}


def _samp_ut3(ws: WorldState, rng: random.Random) -> dict[str, Any] | None:
    return {"window_days": rng.choice(_window_days_choices())} if ws.consumption else None


def _samp_ut4(ws: WorldState, rng: random.Random) -> dict[str, Any] | None:
    pids = _products_with_licenses(ws)
    if not pids:
        return None
    return {"product_id": rng.choice(pids), "window_days": rng.choice(_window_days_choices())}


def _samp_ut5(ws: WorldState, rng: random.Random) -> dict[str, Any] | None:
    pids_with_both = sorted(set(_products_with_licenses(ws)) & set(_products_with_consumption(ws)))
    if not pids_with_both:
        return None
    return {
        "product_id": rng.choice(pids_with_both),
        "window_days": rng.choice(_window_days_choices()),
    }


# ── Template registry ────────────────────────────────────────────────────────

TEMPLATES: list[TaskTemplate] = [
    # spend_visibility — T1
    TaskTemplate(
        template_id="SV1",
        intent="spend_visibility",
        tier=1,
        question_template="What is the monthly license spend for cost center {cc} as of {T}?",
        answer_type="float",
        resolver_ref="spend_by_cost_center",
        sampler=_samp_sv1,
    ),
    TaskTemplate(
        template_id="SV2",
        intent="spend_visibility",
        tier=1,
        question_template="How many active users are in cost center {cc} as of {T}?",
        answer_type="int",
        resolver_ref="active_user_count_cc",
        sampler=_samp_sv1,
    ),
    TaskTemplate(
        template_id="SV3",
        intent="spend_visibility",
        tier=1,
        question_template="What is the current unit price for product {product_id} as of {T}?",
        answer_type="float",
        resolver_ref="product_price",
        sampler=_samp_sv3,
    ),
    # spend_visibility — T2
    TaskTemplate(
        template_id="SV4",
        intent="spend_visibility",
        tier=2,
        question_template="What is the total monthly license spend across all departments as of {T}?",
        answer_type="float",
        resolver_ref="total_monthly_spend",
        sampler=_samp_sv4,
    ),
    TaskTemplate(
        template_id="SV5",
        intent="spend_visibility",
        tier=2,
        question_template="Which cost center has the highest monthly license spend as of {T}?",
        answer_type="str",
        resolver_ref="top_spending_cost_center",
        sampler=_samp_sv5,
    ),
    TaskTemplate(
        template_id="SV6",
        intent="spend_visibility",
        tier=2,
        question_template=(
            "How many cost centers have monthly license spend above ${threshold} as of {T}?"
        ),
        answer_type="int",
        resolver_ref="cost_centers_above_threshold",
        sampler=_samp_sv6,
    ),
    # savings_opportunity — T1
    TaskTemplate(
        template_id="SO1",
        intent="savings_opportunity",
        tier=1,
        question_template=(
            "How many licenses for product {product_id} have been unused for more than "
            "{idle_days} days as of {T}?"
        ),
        answer_type="int",
        resolver_ref="idle_license_count",
        sampler=_samp_so1,
    ),
    TaskTemplate(
        template_id="SO2",
        intent="savings_opportunity",
        tier=1,
        question_template="How many purchased licenses are currently unassigned as of {T}?",
        answer_type="int",
        resolver_ref="unassigned_license_count",
        sampler=_samp_sv4,  # just needs ws.users to be non-empty
    ),
    # savings_opportunity — T2
    TaskTemplate(
        template_id="SO3",
        intent="savings_opportunity",
        tier=2,
        question_template=(
            "How many assigned licenses in cost center {cc} have had no usage "
            "in the last {idle_days} days as of {T}?"
        ),
        answer_type="int",
        resolver_ref="idle_license_count_cc",
        sampler=_samp_so3,
    ),
    TaskTemplate(
        template_id="SO4",
        intent="savings_opportunity",
        tier=2,
        question_template=(
            "What is the total monthly cost of licenses that have been idle for more than "
            "{idle_days} days as of {T}?"
        ),
        answer_type="float",
        resolver_ref="idle_license_cost",
        sampler=_samp_so4,
    ),
    TaskTemplate(
        template_id="SO5",
        intent="savings_opportunity",
        tier=2,
        question_template=(
            "How many license reclamations (unassignments) occurred "
            "in the last {days_back} days as of {T}?"
        ),
        answer_type="int",
        resolver_ref="license_reclaim_count",
        sampler=_samp_so5,
    ),
    # criticality — T1
    TaskTemplate(
        template_id="CR1",
        intent="criticality",
        tier=1,
        question_template=(
            "How many users were offboarded in the last {days_back} days as of {T}?"
        ),
        answer_type="int",
        resolver_ref="offboard_count_window",
        sampler=_samp_cr1,
    ),
    # criticality — T2
    TaskTemplate(
        template_id="CR2",
        intent="criticality",
        tier=2,
        question_template=(
            "How many license reassignments occurred in the last {days_back} days as of {T}?"
        ),
        answer_type="int",
        resolver_ref="reassignment_count",
        sampler=_samp_cr2,
    ),
    # criticality — T3 (contract document required)
    TaskTemplate(
        template_id="CR3",
        intent="criticality",
        tier=3,
        question_template=("Is contract {contract_id} currently active or expired as of {T}?"),
        answer_type="str",
        resolver_ref="contract_status",
        sampler=_samp_cr_contract,
    ),
    TaskTemplate(
        template_id="CR4",
        intent="criticality",
        tier=3,
        question_template="How many months remain on contract {contract_id} as of {T}?",
        answer_type="float",
        resolver_ref="contract_months_remaining",
        sampler=_samp_cr_contract,
    ),
    TaskTemplate(
        template_id="CR5",
        intent="criticality",
        tier=3,
        question_template="Which vendor is named in contract {contract_id} as of {T}?",
        answer_type="str",
        resolver_ref="contract_vendor",
        sampler=_samp_cr_contract,
    ),
    TaskTemplate(
        template_id="CR6",
        intent="criticality",
        tier=3,
        question_template=(
            "What is the estimated annual value stated in contract {contract_id} as of {T}?"
        ),
        answer_type="float",
        resolver_ref="contract_annual_value",
        sampler=_samp_cr_contract,
    ),
    # utilization — T1
    TaskTemplate(
        template_id="UT1",
        intent="utilization",
        tier=1,
        question_template=(
            "What is the total number of API calls for product {product_id} "
            "in the last {window_days} days as of {T}?"
        ),
        answer_type="int",
        resolver_ref="product_api_calls",
        sampler=_samp_ut_product_consumption,
    ),
    TaskTemplate(
        template_id="UT2",
        intent="utilization",
        tier=1,
        question_template=(
            "How many distinct users actively used product {product_id} "
            "in the last {window_days} days as of {T}?"
        ),
        answer_type="int",
        resolver_ref="product_active_users",
        sampler=_samp_ut_product_consumption,
    ),
    # utilization — T2
    TaskTemplate(
        template_id="UT3",
        intent="utilization",
        tier=2,
        question_template=(
            "Which product had the most session minutes in the last {window_days} days as of {T}?"
        ),
        answer_type="str",
        resolver_ref="top_product_by_session_minutes",
        sampler=_samp_ut3,
    ),
    TaskTemplate(
        template_id="UT4",
        intent="utilization",
        tier=2,
        question_template=(
            "How many assigned licenses for product {product_id} had zero API usage "
            "in the last {window_days} days as of {T}?"
        ),
        answer_type="int",
        resolver_ref="zero_usage_license_count",
        sampler=_samp_ut4,
    ),
    # utilization — T3 (license efficiency requires cross-referencing contract seat entitlement)
    TaskTemplate(
        template_id="UT5",
        intent="utilization",
        tier=3,
        question_template=(
            "According to contract records, what fraction of licensed seats for product "
            "{product_id} were actively used (had at least one API call) in the last "
            "{window_days} days as of {T}?"
        ),
        answer_type="float",
        resolver_ref="license_efficiency",
        sampler=_samp_ut5,
    ),
]

# Convenience lookup
TEMPLATES_BY_TIER: dict[int, list[TaskTemplate]] = {1: [], 2: [], 3: []}
for _tmpl in TEMPLATES:
    TEMPLATES_BY_TIER[_tmpl.tier].append(_tmpl)
