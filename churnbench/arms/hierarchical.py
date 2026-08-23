"""Arm 3 — Hierarchical supervisor-worker decomposition (Protocol-H faithful baseline).

Architecture:
  Supervisor (LLM) receives the question + capability cards, dispatches workers.
  SQL / Mongo / SaaS workers query LIVE data at T — no caching, no staleness.
  Docs worker owns its own ChromaDB index over the contract corpus, built at T_prime
  — the only source of partial staleness in this arm.

  Round 1:  Supervisor → dispatch → workers run in parallel → Synthesis.
  Round 2:  If Synthesis returns "NEED_MORE:", re-dispatch a targeted follow-up.
  Max 2 dispatch rounds per task.

This is the strong baseline.  It should beat NaiveArm (less noise) and ClassicRagArm
(live data for SQL/Mongo/SaaS tiers) while remaining below the grounding arm because
docs staleness still leaks through.

Cost decomposition in the paper comes from ArmResult.trace — each worker records its
own input_tokens/output_tokens so per-worker attribution is exact.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from typing import Any

import chromadb
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import text

from churnbench.arms.base import ArmResult, BaseArm, FabricConfig, active_model, cost_usd, llm
from churnbench.arms.classic_rag import _chunks_docs, _rough_token_count
from churnbench.arms.prompts import (
    DOCS_WORKER_PROMPT,
    MONGO_SUMMARIZE_PROMPT,
    MONGO_WORKER_PROMPT,
    SAAS_WORKER_PROMPT,
    SQL_SUMMARIZE_PROMPT,
    SQL_WORKER_PROMPT,
    SUPERVISOR_PROMPT,
    SYNTHESIS_PROMPT,
    system_prompt,
)
from churnbench.eval.parsing import parse
from churnbench.fabric.saas_client import SaasClient
from churnbench.tasks.schema import Task

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_DOCS_TOP_K = 5


@dataclass
class _WorkerResult:
    """Internal per-worker result — aggregated into ArmResult by answer()."""

    worker: str
    sub_query: str
    response: str
    input_tokens: int
    output_tokens: int
    tool_calls: list[dict[str, Any]]
    source_ts: str  # "live" for SQL/Mongo/SaaS; T_prime ISO string for docs


# ── LLM output parsers ────────────────────────────────────────────────────────


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


def _parse_dispatch_json(raw: str) -> dict[str, Any]:
    """Robustly parse supervisor dispatch JSON.  Falls back to empty plan on error."""
    raw = raw.strip()
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    return {"assignments": [], "direct_answer": None}


def _parse_sql(raw: str) -> str:
    """Extract SQL from possibly markdown-fenced LLM output."""
    raw = raw.strip()
    m = re.search(r"```(?:sql)?\s*\n(.*?)\n```", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return raw


def _parse_mongo_query(raw: str) -> dict[str, Any]:
    """Parse MongoDB find query JSON from LLM output."""
    raw = raw.strip()
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    return {"collection": "users", "filter": {}, "projection": None, "limit": 20}


def _parse_saas_paths(raw: str) -> list[str]:
    """Extract list of API paths from LLM output."""
    raw = raw.strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return [str(p) for p in result]
    except json.JSONDecodeError:
        pass
    lines = [ln.strip().strip("\"'[]") for ln in raw.split("\n") if ln.strip().startswith("/")]
    return lines if lines else ["/tickets"]


def _format_worker_outputs(worker_results: list[_WorkerResult]) -> str:
    """Render worker results for inclusion in round-2 supervisor prompt."""
    parts = []
    for wr in worker_results:
        parts.append(f"[{wr.worker.upper()}] Query: {wr.sub_query}\nResult: {wr.response[:500]}")
    return "\n\n".join(parts)


# ── Worker functions ──────────────────────────────────────────────────────────


def _sql_worker(sub_query: str, engine: Any, lm: Any) -> _WorkerResult:
    """Generate SQL, execute, retry once on error, summarize results."""
    tool_calls: list[dict[str, Any]] = []
    total_inp = total_out = 0

    # LLM call 1: generate SQL
    resp1 = lm.invoke([SystemMessage(content=SQL_WORKER_PROMPT), HumanMessage(content=sub_query)])
    meta1 = getattr(resp1, "usage_metadata", {}) or {}
    total_inp += int(meta1.get("input_tokens", 0))
    total_out += int(meta1.get("output_tokens", 0))
    sql_query = _parse_sql(_extract_text(resp1.content))

    # Execute
    t0 = time.perf_counter()
    rows_text = ""
    sql_error: str | None = None
    try:
        with engine.connect() as conn:
            res = conn.execute(text(sql_query))
            rows = res.fetchmany(50)
            cols = list(res.keys())
        if rows:
            header = " | ".join(str(c) for c in cols)
            body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
            rows_text = f"{header}\n{body}\n({len(rows)} rows)"
        else:
            rows_text = "(0 rows)"
    except Exception as exc:
        sql_error = str(exc)
    tool_calls.append(
        {
            "tool": "sql_query",
            "target": sql_query[:80],
            "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    )

    # Retry once on SQL error
    if sql_error is not None:
        retry_msg = (
            f"Query:\n{sql_query}\n\nError: {sql_error}\n\n"
            f"Original question: {sub_query}\n\nRewrite the query to fix the error."
        )
        resp_r = lm.invoke(
            [SystemMessage(content=SQL_WORKER_PROMPT), HumanMessage(content=retry_msg)]
        )
        meta_r = getattr(resp_r, "usage_metadata", {}) or {}
        total_inp += int(meta_r.get("input_tokens", 0))
        total_out += int(meta_r.get("output_tokens", 0))
        sql2 = _parse_sql(_extract_text(resp_r.content))

        t1 = time.perf_counter()
        try:
            with engine.connect() as conn2:
                res2 = conn2.execute(text(sql2))
                rows2 = res2.fetchmany(50)
                cols2 = list(res2.keys())
            if rows2:
                header2 = " | ".join(str(c) for c in cols2)
                body2 = "\n".join(" | ".join(str(v) for v in row) for row in rows2)
                rows_text = f"{header2}\n{body2}\n({len(rows2)} rows)"
            else:
                rows_text = "(0 rows)"
        except Exception as exc2:
            rows_text = f"(query failed after retry: {exc2})"
        tool_calls.append(
            {
                "tool": "sql_query_retry",
                "target": sql2[:80],
                "duration_ms": round((time.perf_counter() - t1) * 1000, 1),
            }
        )

    # LLM call 2 (or 3): summarize results
    if rows_text and not rows_text.startswith("(query failed"):
        resp3 = lm.invoke(
            [
                SystemMessage(content=SQL_SUMMARIZE_PROMPT),
                HumanMessage(content=f"Question: {sub_query}\n\nSQL results:\n{rows_text}"),
            ]
        )
        meta3 = getattr(resp3, "usage_metadata", {}) or {}
        total_inp += int(meta3.get("input_tokens", 0))
        total_out += int(meta3.get("output_tokens", 0))
        summary = _extract_text(resp3.content).strip()
    else:
        summary = rows_text or "No SQL results."

    return _WorkerResult(
        worker="sql",
        sub_query=sub_query,
        response=summary,
        input_tokens=total_inp,
        output_tokens=total_out,
        tool_calls=tool_calls,
        source_ts="live",
    )


def _mongo_worker(sub_query: str, db: Any, lm: Any) -> _WorkerResult:
    """Generate MongoDB find query, execute, retry once on error, summarize."""
    tool_calls: list[dict[str, Any]] = []
    total_inp = total_out = 0

    # LLM call 1: generate find query
    resp1 = lm.invoke([SystemMessage(content=MONGO_WORKER_PROMPT), HumanMessage(content=sub_query)])
    meta1 = getattr(resp1, "usage_metadata", {}) or {}
    total_inp += int(meta1.get("input_tokens", 0))
    total_out += int(meta1.get("output_tokens", 0))
    qd = _parse_mongo_query(_extract_text(resp1.content))

    coll_name = str(qd.get("collection", "users"))
    filter_doc: dict[str, Any] = qd.get("filter") or {}
    projection = qd.get("projection") or None
    limit = int(qd.get("limit", 20))

    # Execute
    t0 = time.perf_counter()
    docs_text = ""
    mongo_error: str | None = None
    try:
        cursor = db[coll_name].find(filter_doc, projection).limit(limit)
        docs = list(cursor)
        if docs:
            docs_text = "\n".join(json.dumps(d, default=str) for d in docs)
            docs_text += f"\n({len(docs)} documents)"
        else:
            docs_text = "(0 documents)"
    except Exception as exc:
        mongo_error = str(exc)
    tool_calls.append(
        {
            "tool": "mongo_find",
            "target": f"{coll_name}[{json.dumps(filter_doc)[:60]}]",
            "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    )

    # Retry once on error
    if mongo_error is not None:
        retry_msg = (
            f"Query: collection={coll_name}, filter={json.dumps(filter_doc)}\n"
            f"Error: {mongo_error}\n\nOriginal question: {sub_query}\n\n"
            "Rewrite the query to fix the error."
        )
        resp_r = lm.invoke(
            [SystemMessage(content=MONGO_WORKER_PROMPT), HumanMessage(content=retry_msg)]
        )
        meta_r = getattr(resp_r, "usage_metadata", {}) or {}
        total_inp += int(meta_r.get("input_tokens", 0))
        total_out += int(meta_r.get("output_tokens", 0))
        qd2 = _parse_mongo_query(_extract_text(resp_r.content))

        t1 = time.perf_counter()
        coll2 = str(qd2.get("collection", coll_name))
        filter2: dict[str, Any] = qd2.get("filter") or {}
        try:
            cursor2 = db[coll2].find(filter2).limit(20)
            docs2 = list(cursor2)
            if docs2:
                docs_text = "\n".join(json.dumps(d, default=str) for d in docs2)
                docs_text += f"\n({len(docs2)} documents)"
            else:
                docs_text = "(0 documents)"
        except Exception as exc2:
            docs_text = f"(mongo query failed after retry: {exc2})"
        tool_calls.append(
            {
                "tool": "mongo_find_retry",
                "target": f"{coll2}[{json.dumps(filter2)[:60]}]",
                "duration_ms": round((time.perf_counter() - t1) * 1000, 1),
            }
        )

    # LLM call 2: summarize
    if docs_text and not docs_text.startswith("(mongo query failed"):
        resp3 = lm.invoke(
            [
                SystemMessage(content=MONGO_SUMMARIZE_PROMPT),
                HumanMessage(
                    content=f"Question: {sub_query}\n\nMongoDB results:\n{docs_text[:2000]}"
                ),
            ]
        )
        meta3 = getattr(resp3, "usage_metadata", {}) or {}
        total_inp += int(meta3.get("input_tokens", 0))
        total_out += int(meta3.get("output_tokens", 0))
        summary = _extract_text(resp3.content).strip()
    else:
        summary = docs_text or "No MongoDB results."

    return _WorkerResult(
        worker="mongo",
        sub_query=sub_query,
        response=summary,
        input_tokens=total_inp,
        output_tokens=total_out,
        tool_calls=tool_calls,
        source_ts="live",
    )


def _saas_worker(sub_query: str, saas: SaasClient, lm: Any) -> _WorkerResult:
    """LLM decides which SaaS endpoints to hit; calls them; returns raw responses."""
    tool_calls: list[dict[str, Any]] = []
    total_inp = total_out = 0

    # LLM call 1: decide paths
    resp1 = lm.invoke([SystemMessage(content=SAAS_WORKER_PROMPT), HumanMessage(content=sub_query)])
    meta1 = getattr(resp1, "usage_metadata", {}) or {}
    total_inp += int(meta1.get("input_tokens", 0))
    total_out += int(meta1.get("output_tokens", 0))
    paths = _parse_saas_paths(_extract_text(resp1.content))

    # Call each path (cap at 5)
    responses: list[str] = []
    for path in paths[:5]:
        t0 = time.perf_counter()
        try:
            raw = saas.raw_get(path)
            responses.append(f"GET {path}: {json.dumps(raw, default=str)[:500]}")
        except Exception as exc:
            responses.append(f"GET {path}: error — {exc}")
        tool_calls.append(
            {
                "tool": "saas_get",
                "target": path,
                "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
        )

    response_text = "\n".join(responses) if responses else "(no SaaS responses)"
    return _WorkerResult(
        worker="saas",
        sub_query=sub_query,
        response=response_text,
        input_tokens=total_inp,
        output_tokens=total_out,
        tool_calls=tool_calls,
        source_ts="live",
    )


def _docs_worker(
    sub_query: str,
    collection: Any,
    embed: Any,
    lm: Any,
    t_prime_iso: str,
) -> _WorkerResult:
    """Embed query, retrieve from T_prime ChromaDB docs index, summarize with LLM."""
    total_inp = total_out = 0

    q_emb: list[list[float]] = embed.encode([sub_query], show_progress_bar=False).tolist()
    n = min(_DOCS_TOP_K, collection.count())
    docs: list[str] = []
    if n > 0:
        results = collection.query(query_embeddings=q_emb, n_results=n, include=["documents"])
        docs = results["documents"][0] if results["documents"] else []

    context = "\n\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs))
    if not context:
        context = "(no contract documents found)"

    resp = lm.invoke(
        [
            SystemMessage(content=DOCS_WORKER_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {sub_query}"),
        ]
    )
    meta = getattr(resp, "usage_metadata", {}) or {}
    total_inp += int(meta.get("input_tokens", 0))
    total_out += int(meta.get("output_tokens", 0))
    summary = _extract_text(resp.content).strip()

    return _WorkerResult(
        worker="docs",
        sub_query=sub_query,
        response=summary,
        input_tokens=total_inp,
        output_tokens=total_out,
        tool_calls=[],
        source_ts=t_prime_iso,
    )


# ── Arm class ─────────────────────────────────────────────────────────────────


class HierarchicalArm(BaseArm):
    """Arm 3: supervisor-worker decomposition — the strong baseline.

    Key differences from the other arms:
    - SQL / Mongo / SaaS workers query LIVE at T: no freshness error on those tiers.
    - Only the docs ChromaDB index is built at T_prime, so partial staleness is possible
      for contract-prose questions but not for operational / warehouse questions.
    - Per-worker cost is tracked in trace for the paper's decomposition table.
    """

    def __init__(
        self,
        *,
        _saas_client: SaasClient | None = None,
        _embed_model: Any = None,
        _llm: Any = None,
    ) -> None:
        """
        _saas_client, _embed_model, _llm: injection points for testing.
        Production code leaves all three as None (defaults applied in setup).
        """
        self._saas_override = _saas_client
        self._embed_override = _embed_model
        self._llm_override = _llm
        self._docs_coll: Any = None
        self._embed: Any = None
        self._engine: Any = None
        self._mongo_db: Any = None
        self._saas: Any = None
        self._lm: Any = None
        self._t_prime_iso: str = ""
        self._embedding_tokens: int = 0
        self._model: str = active_model()

    def setup(self, config: FabricConfig, T_prime: date) -> None:
        """Open DB connections and build the docs ChromaDB index at T_prime."""
        from pymongo import MongoClient
        from sentence_transformers import SentenceTransformer
        from sqlalchemy import create_engine as _ce

        self._model = active_model()
        self._lm = self._llm_override or llm(self._model)
        self._t_prime_iso = T_prime.isoformat()

        self._engine = _ce(config.pg_url, future=True)
        mongo_client: MongoClient[Any] = MongoClient(config.mongo_url)
        self._mongo_db = mongo_client["sam_ops"]
        self._saas = self._saas_override or SaasClient(config.saas_base_url)

        embed = self._embed_override or SentenceTransformer(_EMBED_MODEL_NAME)
        self._embed = embed

        # Docs-only ChromaDB index built at T_prime — partial staleness surface
        client = chromadb.EphemeralClient()
        self._docs_coll = client.create_collection(
            name="hier_docs",
            metadata={"T_prime": T_prime.isoformat(), "hnsw:space": "cosine"},
        )
        chunks = _chunks_docs(config.docs_dir)
        if chunks:
            texts = [c["text"] for c in chunks]
            embeddings: list[list[float]] = embed.encode(
                texts, show_progress_bar=False, batch_size=64
            ).tolist()
            self._embedding_tokens = _rough_token_count(texts)
            self._docs_coll.add(
                embeddings=embeddings,
                documents=texts,
                ids=[f"h{i}" for i in range(len(chunks))],
                metadatas=[c["meta"] for c in chunks],
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _supervisor_dispatch(
        self,
        question: str,
        prior_context: str = "",
        missing_info: str = "",
        round_num: int = 1,
    ) -> tuple[dict[str, Any], int, int]:
        """Call the supervisor LLM.  Returns (plan_dict, input_tokens, output_tokens)."""
        human_parts = [question]
        if prior_context:
            human_parts.append(f"\nPrevious worker outputs:\n{prior_context}")
        if missing_info:
            human_parts.append(f"\nMissing information: {missing_info}")

        response = self._lm.invoke(
            [SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content="\n".join(human_parts))]
        )
        meta = getattr(response, "usage_metadata", {}) or {}
        inp = int(meta.get("input_tokens", 0))
        out = int(meta.get("output_tokens", 0))
        plan = _parse_dispatch_json(_extract_text(response.content))
        return plan, inp, out

    def _dispatch_one(self, assignment: dict[str, str]) -> _WorkerResult:
        """Route one assignment to the appropriate worker function."""
        worker = assignment.get("worker", "")
        query = assignment.get("query", "")
        if worker == "sql":
            return _sql_worker(query, self._engine, self._lm)
        if worker == "mongo":
            return _mongo_worker(query, self._mongo_db, self._lm)
        if worker == "saas":
            return _saas_worker(query, self._saas, self._lm)
        if worker == "docs":
            return _docs_worker(query, self._docs_coll, self._embed, self._lm, self._t_prime_iso)
        return _WorkerResult(
            worker=worker,
            sub_query=query,
            response=f"unknown worker type: {worker}",
            input_tokens=0,
            output_tokens=0,
            tool_calls=[],
            source_ts="live",
        )

    def _run_parallel(
        self,
        assignments: list[dict[str, str]],
        _dispatch_fn: Callable[[dict[str, str]], _WorkerResult] | None = None,
    ) -> list[_WorkerResult]:
        """Dispatch all assignments concurrently.  _dispatch_fn overrides for tests."""
        if not assignments:
            return []
        dispatch = _dispatch_fn or self._dispatch_one
        results: list[_WorkerResult] = []
        with ThreadPoolExecutor(max_workers=min(len(assignments), 4)) as executor:
            futures = {executor.submit(dispatch, a): a for a in assignments}
            for future in as_completed(futures):
                a = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        _WorkerResult(
                            worker=a.get("worker", "unknown"),
                            sub_query=a.get("query", ""),
                            response=f"worker error: {exc}",
                            input_tokens=0,
                            output_tokens=0,
                            tool_calls=[],
                            source_ts="live",
                        )
                    )
        return results

    def _synthesize(
        self,
        question: str,
        worker_results: list[_WorkerResult],
        answer_type: str,
    ) -> tuple[str, int, int]:
        """Single synthesis LLM call.  Returns (raw_answer, input_tokens, output_tokens)."""
        worker_block = _format_worker_outputs(worker_results)
        sys = system_prompt(answer_type) + "\n\n" + SYNTHESIS_PROMPT
        human = f"Worker outputs:\n{worker_block}\n\nQuestion: {question}"
        response = self._lm.invoke([SystemMessage(content=sys), HumanMessage(content=human)])
        meta = getattr(response, "usage_metadata", {}) or {}
        inp = int(meta.get("input_tokens", 0))
        out = int(meta.get("output_tokens", 0))
        return _extract_text(response.content).strip(), inp, out

    # ── Public interface ──────────────────────────────────────────────────────

    def answer(self, task: Task) -> ArmResult:
        if self._lm is None:
            raise RuntimeError("HierarchicalArm.setup() must be called before answer()")

        t0 = time.perf_counter()
        trace: list[dict[str, Any]] = []
        all_tool_calls: list[dict[str, Any]] = []
        total_inp = total_out = 0

        # ── Round 1: Supervisor ───────────────────────────────────────────────
        plan, sup_inp, sup_out = self._supervisor_dispatch(task.question_text, round_num=1)
        total_inp += sup_inp
        total_out += sup_out
        assignments: list[dict[str, str]] = plan.get("assignments") or []
        trace.append(
            {
                "role": "supervisor_round_1",
                "assignments": [a.get("worker") for a in assignments],
                "input_tokens": sup_inp,
                "output_tokens": sup_out,
            }
        )

        # Supervisor answered directly — skip workers
        direct = plan.get("direct_answer")
        if direct is not None:
            answer_raw = str(direct)
            parsed = parse(answer_raw, task.answer_type)
            trace.append({"role": "direct_answer", "content": answer_raw})
            return ArmResult(
                task_id=task.task_id,
                answer_raw=answer_raw,
                answer_parsed=parsed.value,
                input_tokens=total_inp,
                output_tokens=total_out,
                embedding_tokens=self._embedding_tokens,
                cost_usd=cost_usd(self._model, total_inp, total_out, self._embedding_tokens),
                latency_s=round(time.perf_counter() - t0, 3),
                tool_calls=all_tool_calls,
                trace=trace,
            )

        # ── Workers (parallel) ────────────────────────────────────────────────
        worker_results = self._run_parallel(assignments)
        for wr in worker_results:
            total_inp += wr.input_tokens
            total_out += wr.output_tokens
            all_tool_calls.extend(wr.tool_calls)
            trace.append(
                {
                    "role": f"{wr.worker}_worker",
                    "query": wr.sub_query,
                    "result": wr.response[:300],
                    "input_tokens": wr.input_tokens,
                    "output_tokens": wr.output_tokens,
                    "source_ts": wr.source_ts,
                }
            )

        # ── Synthesis round 1 ─────────────────────────────────────────────────
        synthesis_raw, syn_inp, syn_out = self._synthesize(
            task.question_text, worker_results, task.answer_type
        )
        total_inp += syn_inp
        total_out += syn_out
        trace.append(
            {
                "role": "synthesis_round_1",
                "content": synthesis_raw[:300],
                "input_tokens": syn_inp,
                "output_tokens": syn_out,
            }
        )

        # ── Round 2 if synthesis requests more information ────────────────────
        if synthesis_raw.startswith("NEED_MORE:"):
            missing = synthesis_raw[len("NEED_MORE:") :].strip()
            prior_ctx = _format_worker_outputs(worker_results)

            plan2, sup2_inp, sup2_out = self._supervisor_dispatch(
                task.question_text,
                prior_context=prior_ctx,
                missing_info=missing,
                round_num=2,
            )
            total_inp += sup2_inp
            total_out += sup2_out
            assignments2: list[dict[str, str]] = plan2.get("assignments") or []
            trace.append(
                {
                    "role": "supervisor_round_2",
                    "assignments": [a.get("worker") for a in assignments2],
                    "input_tokens": sup2_inp,
                    "output_tokens": sup2_out,
                }
            )

            if assignments2:
                worker_results2 = self._run_parallel(assignments2)
                for wr2 in worker_results2:
                    total_inp += wr2.input_tokens
                    total_out += wr2.output_tokens
                    all_tool_calls.extend(wr2.tool_calls)
                    trace.append(
                        {
                            "role": f"{wr2.worker}_worker_r2",
                            "query": wr2.sub_query,
                            "result": wr2.response[:300],
                            "input_tokens": wr2.input_tokens,
                            "output_tokens": wr2.output_tokens,
                            "source_ts": wr2.source_ts,
                        }
                    )
                all_results = worker_results + worker_results2
            else:
                all_results = worker_results

            synthesis_raw, syn2_inp, syn2_out = self._synthesize(
                task.question_text, all_results, task.answer_type
            )
            total_inp += syn2_inp
            total_out += syn2_out
            trace.append(
                {
                    "role": "synthesis_final",
                    "content": synthesis_raw[:300],
                    "input_tokens": syn2_inp,
                    "output_tokens": syn2_out,
                }
            )

        latency_s = round(time.perf_counter() - t0, 3)
        parsed = parse(synthesis_raw, task.answer_type)
        c = cost_usd(self._model, total_inp, total_out, self._embedding_tokens)

        return ArmResult(
            task_id=task.task_id,
            answer_raw=synthesis_raw,
            answer_parsed=parsed.value,
            input_tokens=total_inp,
            output_tokens=total_out,
            embedding_tokens=self._embedding_tokens,
            cost_usd=c,
            latency_s=latency_s,
            tool_calls=all_tool_calls,
            trace=trace,
        )

    def teardown(self) -> None:
        if self._saas is not None and self._saas is not self._saas_override:
            try:
                self._saas.close()
            except Exception:
                pass
        if self._engine is not None:
            try:
                self._engine.dispose()
            except Exception:
                pass
        self._docs_coll = None
        self._embed = None
