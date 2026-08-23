"""TaskSetGenerator: deterministic evaluation task set from a ledger at timestamp T."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from churnbench.ledger.fold import fold_events
from churnbench.ledger.ledger import Ledger
from churnbench.tasks.resolver import LedgerResolver
from churnbench.tasks.schema import Task, save_tasks
from churnbench.tasks.templates import TEMPLATES_BY_TIER

# Each template may contribute at most this many zero-gold tasks to the set.
# This bounds the trivial-arm score: with 22 templates, worst-case zero-gold
# fraction = PER_TEMPLATE_ZERO_QUOTA * 22 / n ≈ 22/180 ≈ 12% — but in
# practice only 1–3 templates ever produce zero, so the real rate is ~1–2%.
PER_TEMPLATE_ZERO_QUOTA = 1


def _is_degenerate(value: Any) -> bool:
    """True if the gold answer is trivially zero/empty.

    Zero counts and empty-string sentinels are flagged so the generator can
    apply the per-template quota.  This does NOT mean the answer is wrong —
    "0 orphaned licenses" is a real answer — but an excess of zeros lets a
    trivial always-zero arm score high.  The quota caps how many such tasks
    each template can contribute.
    """
    if value is None:
        return True
    if isinstance(value, str) and value in {"", "not_found"}:
        return True
    if isinstance(value, (int, float)) and value == 0:
        return True
    return False


@dataclass
class TaskSetGenerator:
    """Generates n evaluation tasks deterministically from a ledger frozen at T.

    Tier balance target: ~40% tier-1, ~40% tier-2, ~20% tier-3.
    Per-template zero-gold quota (PER_TEMPLATE_ZERO_QUOTA) prevents any single
    structurally-zero template from inflating the trivial-arm score.
    """

    ledger: Ledger
    T: date
    seed: int
    n: int = 180

    def generate(self) -> list[Task]:
        rng = random.Random(self.seed)
        ws = fold_events(self.ledger.events_through(self.T))
        resolver = LedgerResolver(self.ledger)

        tasks: list[Task] = []
        # Per-template zero-gold counter — quota enforced per template_id
        tmpl_zero_count: dict[str, int] = defaultdict(int)
        attempts = 0
        max_attempts = self.n * 25

        while len(tasks) < self.n and attempts < max_attempts:
            attempts += 1

            # Weighted tier selection: 40/40/20
            tier = rng.choices([1, 2, 3], weights=[40, 40, 20])[0]
            pool = TEMPLATES_BY_TIER.get(tier, [])
            if not pool:
                continue

            tmpl = rng.choice(pool)
            params = tmpl.sample_params(ws, rng)
            if params is None:
                continue

            task = Task(
                task_id=f"task_{len(tasks):04d}",
                template_id=tmpl.template_id,
                intent=tmpl.intent,
                tier=tmpl.tier,
                question_text=tmpl.question_template.format(**params, T=self.T.isoformat()),
                params=params,
                T=self.T,
                answer_type=tmpl.answer_type,
                resolver_ref=tmpl.resolver_ref,
            )

            gold = task.gold(resolver, at=self.T)
            if _is_degenerate(gold.value):
                if tmpl_zero_count[tmpl.template_id] >= PER_TEMPLATE_ZERO_QUOTA:
                    continue  # this template has exhausted its zero-gold quota
                tmpl_zero_count[tmpl.template_id] += 1
                # Fall through: include within quota

            tasks.append(task)

        return tasks

    def generate_and_save(self, out_dir: Path) -> list[Task]:
        tasks = self.generate()
        save_tasks(tasks, out_dir / "tasks.jsonl")
        return tasks
