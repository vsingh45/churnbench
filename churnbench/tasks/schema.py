"""Task schema: GoldAnswer and Task dataclasses, plus JSONL persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from churnbench.tasks.resolver import LedgerResolver

AnswerType = Literal["float", "int", "str", "list[str]"]


@dataclass
class GoldAnswer:
    value: Any
    answer_type: str
    resolver_ref: str


@dataclass
class Task:
    task_id: str
    template_id: str
    intent: str
    tier: int
    question_text: str
    params: dict[str, Any]
    T: date
    answer_type: str
    resolver_ref: str

    def gold(self, resolver: "LedgerResolver", at: date) -> GoldAnswer:
        """Compute ground-truth answer at an arbitrary timestamp.

        Pass at=T for the canonical gold; pass at=T′ < T to measure what the
        world looked like when a stale cache was built — enabling freshness-error
        attribution without grader-model contamination.
        """
        method = getattr(resolver, self.resolver_ref)
        value: Any = method(**self.params, T=at)
        return GoldAnswer(value=value, answer_type=self.answer_type, resolver_ref=self.resolver_ref)

    def to_jsonl_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "template_id": self.template_id,
            "intent": self.intent,
            "tier": self.tier,
            "question_text": self.question_text,
            "params": self.params,
            "T": self.T.isoformat(),
            "answer_type": self.answer_type,
            "resolver_ref": self.resolver_ref,
        }


def save_tasks(tasks: list[Task], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for task in tasks:
            f.write(json.dumps(task.to_jsonl_record()) + "\n")


def load_tasks(path: Path) -> list[Task]:
    tasks: list[Task] = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            tasks.append(
                Task(
                    task_id=d["task_id"],
                    template_id=d["template_id"],
                    intent=d["intent"],
                    tier=d["tier"],
                    question_text=d["question_text"],
                    params=d["params"],
                    T=date.fromisoformat(d["T"]),
                    answer_type=d["answer_type"],
                    resolver_ref=d["resolver_ref"],
                )
            )
    return tasks
