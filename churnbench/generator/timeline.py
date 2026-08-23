"""ChurnBench timeline simulator.

Runs a discrete-day simulation of a SAM environment and emits ledger events
for every mutation. Fully deterministic given (seed, config). All fabric
projections (Postgres, Mongo, docs, API state) are downstream of this ledger.

Design invariants:
  1. Nothing here reads or writes the fabric. It only appends to the ledger.
  2. Every random draw goes through a single seeded numpy Generator.
  3. Each day's mutations are ordered deterministically by (kind, entity_id).

Parameters exposed to the paper:
  - drift_rate:  mean mutations per day (Poisson lambda)
  - drift_skew:  fraction of mutations concentrated in the top-decile entities
                 (models the real-world reality that a few products/users churn hardest)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from churnbench.ledger.ledger import EventKind, Ledger


# ─── Configuration ──────────────────────────────────────────────────────────


@dataclass
class TimelineConfig:
    start: date = date(2024, 1, 1)
    days: int = 180
    seed: int = 42

    # Initial population
    n_vendors: int = 12
    n_products: int = 40
    n_cost_centers: int = 18
    n_users_initial: int = 400
    n_contracts_initial: int = 25

    # Drift dynamics (per day means; Poisson-distributed)
    hires_per_day: float = 1.8
    offboards_per_day: float = 1.2
    reassignments_per_day: float = 3.5
    price_changes_per_day: float = 0.15
    contract_renewals_per_day: float = 0.20
    cost_center_moves_per_day: float = 0.6

    # Consumption events (denser stream; affects warehouse fact table size)
    consumption_events_per_day: int = 300

    # Skew: probability mass on top-decile hot entities
    hot_entity_skew: float = 0.60


# ─── State the simulator carries during a run ───────────────────────────────


@dataclass
class _SimState:
    rng: np.random.Generator
    day: date
    active_user_ids: list[str] = field(default_factory=list)
    active_license_ids: list[str] = field(default_factory=list)
    product_ids: list[str] = field(default_factory=list)
    contract_ids: list[str] = field(default_factory=list)
    cost_center_ids: list[str] = field(default_factory=list)
    # license_id -> user_ext_id currently holding it (None if unassigned)
    license_holder: dict[str, str | None] = field(default_factory=dict)
    # product_id -> current unit price
    product_price: dict[str, float] = field(default_factory=dict)
    next_user_n: int = 0
    next_license_n: int = 0
    next_contract_n: int = 0


# ─── The simulator ──────────────────────────────────────────────────────────


class TimelineSimulator:
    def __init__(self, cfg: TimelineConfig, ledger: Ledger) -> None:
        self.cfg = cfg
        self.ledger = ledger

    def run(self) -> None:
        state = _SimState(rng=np.random.default_rng(self.cfg.seed), day=self.cfg.start)
        self._seed_world(state)
        for i in range(self.cfg.days):
            state.day = self.cfg.start + timedelta(days=i)
            self._step_day(state)

    # ── Seeding (day 0) ───────────────────────────────────────────────────

    def _seed_world(self, s: _SimState) -> None:
        cfg = self.cfg
        # Cost centers
        for i in range(cfg.n_cost_centers):
            cc_id = f"cc_{i:03d}"
            s.cost_center_ids.append(cc_id)

        # Products (with initial prices)
        for i in range(cfg.n_products):
            pid = f"prd_{i:04d}"
            s.product_ids.append(pid)
            price = float(round(s.rng.uniform(20, 500), 2))
            s.product_price[pid] = price
            self.ledger.append(
                s.day,
                EventKind.PRICE_CHANGED,
                "product",
                pid,
                {"unit_price_usd": price, "reason": "initial"},
            )

        # Contracts
        for i in range(cfg.n_contracts_initial):
            cid = f"ctr_{i:04d}"
            s.contract_ids.append(cid)
            self.ledger.append(
                s.day,
                EventKind.CONTRACT_SIGNED,
                "contract",
                cid,
                {
                    "vendor_idx": int(s.rng.integers(0, cfg.n_vendors)),
                    "term_months": int(s.rng.choice([12, 24, 36])),
                },
            )
        s.next_contract_n = cfg.n_contracts_initial

        # Users
        for _ in range(cfg.n_users_initial):
            uid = self._new_user(s, hired_at=cfg.start)

        # Seed initial license assignments — half the users get a license
        for uid in s.active_user_ids[: cfg.n_users_initial // 2]:
            pid = self._pick_product(s)
            self._issue_license(s, uid, pid)

    # ── Per-day step ──────────────────────────────────────────────────────

    def _step_day(self, s: _SimState) -> None:
        cfg = self.cfg
        # Poisson draws for each mutation stream
        n_hires = int(s.rng.poisson(cfg.hires_per_day))
        n_offboards = int(s.rng.poisson(cfg.offboards_per_day))
        n_reassigns = int(s.rng.poisson(cfg.reassignments_per_day))
        n_price_chg = int(s.rng.poisson(cfg.price_changes_per_day))
        n_renewals = int(s.rng.poisson(cfg.contract_renewals_per_day))
        n_cc_moves = int(s.rng.poisson(cfg.cost_center_moves_per_day))

        for _ in range(n_hires):
            self._new_user(s, hired_at=s.day)

        for _ in range(n_offboards):
            self._offboard_random_user(s)

        for _ in range(n_reassigns):
            self._reassign_random_license(s)

        for _ in range(n_price_chg):
            self._change_random_price(s)

        for _ in range(n_renewals):
            self._renew_random_contract(s)

        for _ in range(n_cc_moves):
            self._move_random_user_cc(s)

        # Consumption stream — append-only, no state mutation
        self._log_consumption(s, n=cfg.consumption_events_per_day)

    # ── Mutation primitives ───────────────────────────────────────────────

    def _new_user(self, s: _SimState, hired_at: date) -> str:
        uid = f"usr_{s.next_user_n:06d}"
        s.next_user_n += 1
        s.active_user_ids.append(uid)
        cc = s.cost_center_ids[int(s.rng.integers(0, len(s.cost_center_ids)))]
        self.ledger.append(
            hired_at,
            EventKind.USER_HIRED,
            "user",
            uid,
            {"cost_center_id": cc},
        )
        return uid

    def _offboard_random_user(self, s: _SimState) -> None:
        if not s.active_user_ids:
            return
        uid = self._pick_hot(s, s.active_user_ids)
        # Any licenses held must be unassigned first (ledger consistency)
        for lic, holder in list(s.license_holder.items()):
            if holder == uid:
                self.ledger.append(
                    s.day,
                    EventKind.LICENSE_UNASSIGNED,
                    "license",
                    lic,
                    {"prev_holder": uid, "reason": "offboard"},
                )
                s.license_holder[lic] = None
        s.active_user_ids.remove(uid)
        self.ledger.append(s.day, EventKind.USER_OFFBOARDED, "user", uid, {})

    def _reassign_random_license(self, s: _SimState) -> None:
        assigned = [lic for lic, h in s.license_holder.items() if h is not None]
        if not assigned or len(s.active_user_ids) < 2:
            return
        lic = self._pick_hot(s, assigned)
        old = s.license_holder[lic]
        candidates = [u for u in s.active_user_ids if u != old]
        new = self._pick_hot(s, candidates)
        s.license_holder[lic] = new
        self.ledger.append(
            s.day,
            EventKind.LICENSE_REASSIGNED,
            "license",
            lic,
            {"from": old, "to": new},
        )

    def _change_random_price(self, s: _SimState) -> None:
        if not s.product_ids:
            return
        pid = self._pick_hot(s, s.product_ids)
        old = s.product_price[pid]
        # ±15% typical price move
        pct = float(s.rng.normal(0.0, 0.15))
        new = max(1.0, round(old * (1.0 + pct), 2))
        s.product_price[pid] = new
        self.ledger.append(
            s.day,
            EventKind.PRICE_CHANGED,
            "product",
            pid,
            {"unit_price_usd": new, "prev_price": old, "pct_change": round(pct, 4)},
        )

    def _renew_random_contract(self, s: _SimState) -> None:
        if not s.contract_ids:
            return
        cid = s.contract_ids[int(s.rng.integers(0, len(s.contract_ids)))]
        self.ledger.append(
            s.day,
            EventKind.CONTRACT_RENEWED,
            "contract",
            cid,
            {"term_months": int(s.rng.choice([12, 24, 36]))},
        )

    def _move_random_user_cc(self, s: _SimState) -> None:
        if not s.active_user_ids:
            return
        uid = s.active_user_ids[int(s.rng.integers(0, len(s.active_user_ids)))]
        new_cc = s.cost_center_ids[int(s.rng.integers(0, len(s.cost_center_ids)))]
        self.ledger.append(
            s.day,
            EventKind.USER_MOVED_COST_CENTER,
            "user",
            uid,
            {"cost_center_id": new_cc},
        )

    def _log_consumption(self, s: _SimState, n: int) -> None:
        for _ in range(n):
            if not s.active_user_ids or not s.product_ids:
                return
            uid = self._pick_hot(s, s.active_user_ids)
            pid = self._pick_hot(s, s.product_ids)
            self.ledger.append(
                s.day,
                EventKind.CONSUMPTION_LOGGED,
                "user",
                uid,
                {
                    "product_id": pid,
                    "session_minutes": int(s.rng.integers(5, 180)),
                    "api_calls": int(s.rng.integers(0, 500)),
                },
            )

    def _issue_license(self, s: _SimState, uid: str, pid: str) -> None:
        lic = f"lic_{s.next_license_n:06d}"
        s.next_license_n += 1
        s.active_license_ids.append(lic)
        s.license_holder[lic] = uid
        self.ledger.append(
            s.day,
            EventKind.LICENSE_PURCHASED,
            "license",
            lic,
            {"product_id": pid, "seats": 1, "unit_price_usd": s.product_price[pid]},
        )
        self.ledger.append(
            s.day,
            EventKind.LICENSE_ASSIGNED,
            "license",
            lic,
            {"to": uid, "product_id": pid},
        )

    def _pick_product(self, s: _SimState) -> str:
        return s.product_ids[int(s.rng.integers(0, len(s.product_ids)))]

    # ── Skewed sampling ───────────────────────────────────────────────────

    def _pick_hot(self, s: _SimState, pool: list[str]) -> str:
        """With prob `hot_entity_skew`, draw from the top decile; else uniform.

        Models the empirical reality that a few products / users / licenses
        churn far more than the rest — the pattern that makes freshness-
        aware caching interesting.
        """
        if not pool:
            raise IndexError("empty pool")
        if s.rng.random() < self.cfg.hot_entity_skew and len(pool) >= 10:
            top = max(1, len(pool) // 10)
            return pool[int(s.rng.integers(0, top))]
        return pool[int(s.rng.integers(0, len(pool)))]
