"""Arm 1 — Naive tool-calling ('tool-call storm' anti-pattern).

setup() does nothing: every question queries the live fabric at T.
The agent has full tool bindings to all four origins (Postgres, MongoDB,
SaaS API, doc corpus) and uses a LangGraph ReAct loop (max 15 rounds) to
gather whatever data it chooses before answering.

Anti-pattern faithfully executed:
  - Practitioners bind tools, give them to an LLM, and trust it to find answers.
  - The agent knows the schemas — real naive deployments discover/document them.
  - 429s are handled by the SaasClient (exponential backoff, 3 retries) — the
    way production code actually deals with rate limits.
  - No caching, no pre-staging: every call hits the live fabric at T.

This means freshness is perfect (no stale cache), but token/latency cost is
high: the LLM may issue many tool calls to locate an answer that a targeted
resolver would compute in one pass.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool, tool
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent
from sqlalchemy import text
from sqlalchemy.engine import Engine

from churnbench.arms.base import ArmResult, BaseArm, FabricConfig, active_model, cost_usd, llm
from churnbench.arms.prompts import SCHEMA_CONTEXT, system_prompt
from churnbench.eval.parsing import parse
from churnbench.fabric.connections import pg_engine
from churnbench.fabric.connections import mongo_db as _mongo_db_factory
from churnbench.fabric.saas_client import SaasClient
from churnbench.tasks.schema import Task

# Maximum ReAct rounds before the graph is forcibly truncated.
# recursion_limit = rounds * 2 + 2 (agent node + tool node per round, +2 buffer)
_MAX_ROUNDS = 15
_RECURSION_LIMIT = _MAX_ROUNDS * 2 + 2


def _extract_text(content: Any) -> str:
    """Flatten AIMessage content (str or list-of-blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        return " ".join(parts)
    return str(content)


def _sum_usage(messages: list[BaseMessage]) -> tuple[int, int]:
    """Sum input and output tokens across all AI messages in a trace."""
    inp = out = 0
    for msg in messages:
        meta = getattr(msg, "usage_metadata", None)
        if meta and isinstance(meta, dict):
            inp += int(meta.get("input_tokens", 0))
            out += int(meta.get("output_tokens", 0))
    return inp, out


def _build_tools(
    engine: Engine,
    mongo_db: Any,  # pymongo.database.Database — not fully typed
    saas: SaasClient,
    docs_dir: Path,
    log: list[dict[str, Any]],
) -> list[BaseTool]:
    """Construct the five tool functions as closures over live connections."""

    @tool
    def sql_query(query: str) -> str:
        """Execute a SELECT query against the Postgres warehouse (sam schema).

        Returns up to 50 rows formatted as pipe-separated text.
        Never use DML (INSERT/UPDATE/DELETE) — SELECT only.
        """
        t0 = time.perf_counter()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query))
                cols = list(result.keys())
                rows = result.fetchmany(50)
            header = " | ".join(cols)
            body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
            out = f"{header}\n{body}\n({len(rows)} rows)"
        except Exception as exc:
            out = f"SQL error: {exc}"
        ms = (time.perf_counter() - t0) * 1_000
        log.append({"tool": "sql_query", "target": query[:80], "duration_ms": round(ms, 1)})
        return out

    @tool
    def mongo_find(collection: str, filter_json: str = "{}") -> str:
        """Query a MongoDB collection in the sam_ops database.

        Collections: users, active_licenses, assignments, entitlements,
        tickets, utilization_current.
        Pass filter_json as a JSON string, e.g. '{"user_id": "usr_000001"}'.
        Returns up to 20 documents.
        """
        t0 = time.perf_counter()
        try:
            flt: dict[str, Any] = json.loads(filter_json)
            docs = list(mongo_db[collection].find(flt, {"_id": 0}).limit(20))
            body = "\n".join(json.dumps(d, default=str) for d in docs)
            out = f"Collection '{collection}' ({len(docs)} docs):\n{body}"
        except Exception as exc:
            out = f"MongoDB error: {exc}"
        ms = (time.perf_counter() - t0) * 1_000
        target = f"{collection}[{filter_json[:40]}]"
        log.append({"tool": "mongo_find", "target": target, "duration_ms": round(ms, 1)})
        return out

    @tool
    def saas_get(path: str) -> str:
        """GET a SaaS API endpoint (350 ms latency, 60 req/min limit, auto-retried).

        Available paths:
          /entitlements/{user_ext_id}
          /tickets  (optional query params: status, product_sku)
          /products/{product_sku}/current-utilization
        """
        t0 = time.perf_counter()
        try:
            data = saas.raw_get(path)
            out = json.dumps(data, default=str)
        except Exception as exc:
            out = f"SaaS error: {exc}"
        ms = (time.perf_counter() - t0) * 1_000
        log.append({"tool": "saas_get", "target": path, "duration_ms": round(ms, 1)})
        return out

    @tool
    def list_documents() -> str:
        """List available contract documents in the docs corpus."""
        t0 = time.perf_counter()
        try:
            names = sorted(p.stem for p in docs_dir.glob("*.md"))
            out = "\n".join(names) if names else "(no documents found)"
        except Exception as exc:
            out = f"Docs error: {exc}"
        ms = (time.perf_counter() - t0) * 1_000
        log.append({"tool": "list_documents", "target": str(docs_dir), "duration_ms": round(ms, 1)})
        return out

    @tool
    def read_document(doc_id: str) -> str:
        """Read the full content of a contract document.

        doc_id is the filename stem (e.g. 'ctr_0001'), not the full path.
        """
        t0 = time.perf_counter()
        try:
            fpath = docs_dir / f"{doc_id}.md"
            out = fpath.read_text()
        except Exception as exc:
            out = f"Doc error: {exc}"
        ms = (time.perf_counter() - t0) * 1_000
        log.append({"tool": "read_document", "target": doc_id, "duration_ms": round(ms, 1)})
        return out

    return [sql_query, mongo_find, saas_get, list_documents, read_document]


class NaiveArm(BaseArm):
    """Arm 1: naive tool-calling — live fabric, no pre-staging.

    setup() does nothing (no index built, no cache warmed).
    answer() runs a LangGraph ReAct loop that may issue up to 15 tool calls
    before producing a final answer.  All four fabric origins are reachable.
    """

    def __init__(
        self,
        *,
        _saas_client: SaasClient | None = None,
        _llm: Any = None,
    ) -> None:
        """
        _saas_client and _llm are injection points for testing.
        Production code leaves both as None (defaults applied in setup).
        """
        self._saas_override = _saas_client
        self._llm_override = _llm
        self._engine: Engine | None = None
        self._mongo: Any = None
        self._saas: SaasClient | None = None
        self._tool_log: list[dict[str, Any]] = []
        self._agent: Any = None  # CompiledGraph
        self._model: str = active_model()

    def setup(self, config: FabricConfig, T_prime: date) -> None:
        """setup() intentionally does nothing visible — that's the anti-pattern.

        Connections are opened here so they can be shared across answer() calls,
        but no caches or indexes are built.  Every answer queries the live fabric.
        """
        self._model = active_model()
        lm = self._llm_override or llm(self._model)

        self._engine = pg_engine(config.pg_url)
        self._mongo = _mongo_db_factory(config.mongo_url)
        self._saas = self._saas_override or SaasClient(config.saas_base_url)

        prompt = f"{SCHEMA_CONTEXT}\n\n{system_prompt('str')}"  # type-agnostic base
        tools = _build_tools(
            self._engine,
            self._mongo,
            self._saas,
            config.docs_dir,
            self._tool_log,
        )
        self._agent = create_react_agent(lm, tools, prompt=prompt)

    def answer(self, task: Task) -> ArmResult:
        if self._agent is None or self._engine is None:
            raise RuntimeError("NaiveArm.setup() must be called before answer()")

        self._tool_log.clear()
        t0 = time.perf_counter()

        # Compose a per-task system prompt that overrides the format instructions
        question_prompt = f"{system_prompt(task.answer_type)}\n\n" f"Question: {task.question_text}"

        messages: list[BaseMessage] = []
        try:
            result = self._agent.invoke(
                {"messages": [("human", question_prompt)]},
                config={"recursion_limit": _RECURSION_LIMIT},
            )
            messages = result.get("messages", [])
        except GraphRecursionError:
            # Agent hit the round limit — use whatever was last said
            pass
        except Exception:
            pass

        latency_s = time.perf_counter() - t0
        inp, out = _sum_usage(messages)

        # Last AI message is the final answer
        answer_raw = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                answer_raw = _extract_text(msg.content)
                break

        parsed = parse(answer_raw, task.answer_type)
        c = cost_usd(self._model, inp, out)

        trace = [
            {
                "role": msg.__class__.__name__,
                "content": _extract_text(msg.content)[:300],
            }
            for msg in messages
        ]

        return ArmResult(
            task_id=task.task_id,
            answer_raw=answer_raw,
            answer_parsed=parsed.value,
            input_tokens=inp,
            output_tokens=out,
            embedding_tokens=0,
            cost_usd=c,
            latency_s=round(latency_s, 3),
            tool_calls=list(self._tool_log),
            trace=trace,
        )

    def teardown(self) -> None:
        if self._saas is not None and self._saas_override is None:
            self._saas.close()
        if self._engine is not None:
            self._engine.dispose()
