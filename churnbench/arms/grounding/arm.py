"""Arm 4 — grounding architecture: the paper's proposed method (§3).

Two-LLM-call flow:
  Call 1 (need-resolution): maps question → {entity_classes, measures, filters}
          constrained against the semantic model vocabulary.
  Deterministic routing: rule-based, no LLM call (Principle 2).
  Query execution: templated SQL where possible (Principle 1), LLM-generated for
                  the long tail; SaaS API for inherently-live; vector search for docs.
  Call 2 (synthesis): final answer assembled from resolved facts.

Ablation flags (one principle each):
  no_freshness_tiers  — bypasses TTL check; all cached data appears fresh regardless of
                        elapsed time (disables Principle 3's tier-aware scheduling)
  no_semantic_model   — skips vocabulary-constrained LLM call; keyword heuristics route
                        instead (disables Principle 2's structured routing)
  no_source_routing   — routes all retrievals through the docs vector index, ignoring
                        entity class metadata (disables Principle 2's routing hierarchy)

Trace fields per retrieval entry:
  entity_class, measure, route, query_method, last_refresh, cache_miss_reason,
  staged_vs_live, duration_ms, result_preview, llm_input_tokens, llm_output_tokens
"""

from __future__ import annotations

import copy
import json
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import chromadb
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import text

from churnbench.arms.base import ArmResult, BaseArm, FabricConfig, active_model, cost_usd, llm
from churnbench.arms.grounding.etl import setup as _etl_setup
from churnbench.arms.grounding.router import (
    FEDERATED_TEMPLATES,
    RouteDecision,
    decide,
    keyword_entity_classes,
)
from churnbench.arms.grounding.semantic_model import (
    ENTITY_REGISTRY,
    MEASURE_TO_ENTITY,
    EntityClass,
)
from churnbench.arms.prompts import SQL_WORKER_PROMPT, system_prompt
from churnbench.eval.parsing import parse
from churnbench.tasks.schema import Task

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_DOCS_TOP_K = 5
_FULL_INDEX_TOP_K = 8


def _extract_text(content: Any) -> str:
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


def _strip_sql_fences(raw: str) -> str:
    m = re.search(r"```(?:sql)?\s*\n(.*?)\n```", raw, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else raw.strip()


def _parse_need_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {"entity_classes": [], "measures": [], "filters": {}}


def _rough_token_count(texts: list[str]) -> int:
    return sum(len(t) // 4 for t in texts)


def _build_need_resolution_prompt(registry: dict[str, EntityClass]) -> str:
    vocab_lines = "\n".join(f"  {name}: measures={ec.measures}" for name, ec in registry.items())
    return f"""\
You are a need-resolution engine for an enterprise analytics system.

Available entity classes and their measures:
{vocab_lines}

Given the analytics question below, identify:
1. Which entity classes are needed (use only the names above)
2. Which specific measures are requested (use only the measure names above)
3. Filter values mentioned in the question (null if absent)

Output ONLY a JSON object — no markdown, no explanation:
{{
  "entity_classes": ["<name>", ...],
  "measures": ["<measure>", ...],
  "filters": {{
    "cost_center": "<cc_id or null>",
    "product_sku": "<sku or null>",
    "product_id": "<pid or null>",
    "user_id": "<user_id or null>",
    "vendor_id": "<vendor_id or null>",
    "contract_id": "<ctr_id or null>"
  }}
}}
"""


def _chunks_docs(docs_dir: Path) -> list[dict[str, Any]]:
    """Produce text chunks from the contract document corpus."""
    chunks: list[dict[str, Any]] = []
    if not docs_dir.exists():
        return chunks
    for md_file in sorted(docs_dir.glob("*.md")):
        text_content = md_file.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in text_content.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs):
            if len(para) > 20:
                chunks.append({"text": para, "meta": {"source": md_file.name, "chunk": i}})
    return chunks


class GroundingArm(BaseArm):
    """Arm 4 — the grounding architecture proposed in §3 of the paper.

    All ablation flags default to False (= the proposed method).
    Flip exactly one flag at a time to produce the three ablations described in §4.
    """

    def __init__(
        self,
        *,
        no_freshness_tiers: bool = False,
        no_semantic_model: bool = False,
        no_source_routing: bool = False,
        _llm: Any = None,
        _staged_engine: Any = None,
        _saas_client: Any = None,
        _embed_model: Any = None,
    ) -> None:
        self.no_freshness_tiers = no_freshness_tiers
        self.no_semantic_model = no_semantic_model
        self.no_source_routing = no_source_routing

        # Test injection hooks
        self._llm_override = _llm
        self._staged_engine_override = _staged_engine
        self._saas_override = _saas_client
        self._embed_override = _embed_model

        self._lm: Any = None
        self._model: str = active_model()
        self._staged_engine: Any = None
        self._pg_engine: Any = None
        self._mongo_db: Any = None
        self._saas: Any = None
        self._embed: Any = None
        self._docs_coll: Any = None  # docs-only ChromaDB collection
        self._full_coll: Any = None  # full-index (no_source_routing ablation)
        self._registry: dict[str, EntityClass] = {}
        self._embedding_tokens: int = 0
        self._t_prime: date = date.min

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def setup(self, config: FabricConfig, T_prime: date) -> None:
        from pymongo import MongoClient
        from sentence_transformers import SentenceTransformer
        from sqlalchemy import create_engine as _ce

        self._model = active_model()
        self._lm = self._llm_override or llm(self._model)
        self._t_prime = T_prime
        self._registry = copy.deepcopy(ENTITY_REGISTRY)

        self._pg_engine = _ce(config.pg_url, future=True)
        mongo_client: MongoClient[Any] = MongoClient(config.mongo_url)
        self._mongo_db = mongo_client["sam_ops"]
        self._saas = self._saas_override

        self._embed = self._embed_override or SentenceTransformer(_EMBED_MODEL_NAME)

        self._staged_engine = self._staged_engine_override or _ce("sqlite:///:memory:", future=True)

        _etl_setup(
            self._registry,
            self._staged_engine,
            pg_engine=self._pg_engine,
            mongo_db=self._mongo_db,
            T_prime=T_prime,
        )

        # Docs-only ChromaDB index for contract_terms
        chroma = chromadb.EphemeralClient()
        self._docs_coll = chroma.create_collection(
            name="grounding_docs",
            metadata={"T_prime": T_prime.isoformat(), "hnsw:space": "cosine"},
        )
        doc_chunks = _chunks_docs(config.docs_dir)
        if doc_chunks:
            texts = [c["text"] for c in doc_chunks]
            embeddings: list[list[float]] = self._embed.encode(
                texts, show_progress_bar=False, batch_size=64
            ).tolist()
            self._embedding_tokens += _rough_token_count(texts)
            self._docs_coll.add(
                embeddings=embeddings,
                documents=texts,
                ids=[f"gd{i}" for i in range(len(doc_chunks))],
                metadatas=[c["meta"] for c in doc_chunks],
            )

        # Full vector index (no_source_routing ablation only)
        if self.no_source_routing:
            self._full_coll = chroma.create_collection(
                name="grounding_full",
                metadata={"T_prime": T_prime.isoformat(), "hnsw:space": "cosine"},
            )
            # Serialize staged SQL tables as natural-language chunks for the full index
            all_chunks = list(doc_chunks)
            all_chunks.extend(self._staged_as_text_chunks())
            if all_chunks:
                full_texts = [c["text"] for c in all_chunks]
                full_emb: list[list[float]] = self._embed.encode(
                    full_texts, show_progress_bar=False, batch_size=64
                ).tolist()
                self._embedding_tokens += _rough_token_count(full_texts)
                self._full_coll.add(
                    embeddings=full_emb,
                    documents=full_texts,
                    ids=[f"gf{i}" for i in range(len(all_chunks))],
                    metadatas=[c["meta"] for c in all_chunks],
                )

    def _staged_as_text_chunks(self) -> list[dict[str, Any]]:
        """Serialize staged SQLite tables as text for the no_source_routing full index."""
        chunks: list[dict[str, Any]] = []
        if self._staged_engine is None:
            return chunks
        table_queries = [
            ("staged_users", "SELECT * FROM staged_users LIMIT 500"),
            ("staged_assignments", "SELECT * FROM staged_assignments LIMIT 500"),
            ("staged_license_purchases", "SELECT * FROM staged_license_purchases LIMIT 500"),
            ("staged_cost_centers", "SELECT * FROM staged_cost_centers LIMIT 100"),
            ("staged_vendors", "SELECT * FROM staged_vendors LIMIT 100"),
        ]
        try:
            with self._staged_engine.connect() as conn:
                for tbl, sql in table_queries:
                    res = conn.execute(text(sql))
                    cols = list(res.keys())
                    for row in res.fetchall():
                        line = ", ".join(f"{c}={v}" for c, v in zip(cols, row))
                        chunks.append({"text": line, "meta": {"source": tbl}})
        except Exception:
            pass
        return chunks

    def teardown(self) -> None:
        if self._pg_engine is not None:
            try:
                self._pg_engine.dispose()
            except Exception:
                pass
        self._docs_coll = None
        self._full_coll = None
        self._embed = None
        self._lm = None

    # ── LLM call 1: need-resolution ───────────────────────────────────────────

    def _resolve_needs(self, question: str) -> tuple[dict[str, Any], int, int]:
        """Map question → entity classes + measures + filters.

        Returns (need_dict, input_tokens, output_tokens).
        no_semantic_model ablation: keyword heuristics, 0 tokens.
        """
        if self.no_semantic_model:
            ec_names = keyword_entity_classes(question)
            return ({"entity_classes": ec_names, "measures": [], "filters": {}}, 0, 0)

        sys_prompt = _build_need_resolution_prompt(self._registry)
        response = self._lm.invoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=question)]
        )
        meta = getattr(response, "usage_metadata", {}) or {}
        inp = int(meta.get("input_tokens", 0))
        out = int(meta.get("output_tokens", 0))
        need = _parse_need_json(_extract_text(response.content))

        # Constrain to known vocabulary
        known_ec = set(self._registry.keys())
        need["entity_classes"] = [e for e in need.get("entity_classes", []) if e in known_ec]
        known_m = set(MEASURE_TO_ENTITY.keys())
        need["measures"] = [m for m in need.get("measures", []) if m in known_m]

        # If measures named but entity_classes empty, infer from MEASURE_TO_ENTITY
        if need["measures"] and not need["entity_classes"]:
            need["entity_classes"] = list({MEASURE_TO_ENTITY[m] for m in need["measures"]})

        if not need["entity_classes"]:
            need["entity_classes"] = ["user_status"]  # graceful fallback

        return need, inp, out

    # ── Deterministic routing + execution ─────────────────────────────────────

    def _execute_retrieval(
        self, need: dict[str, Any], T: date, question: str = ""
    ) -> tuple[str, list[dict[str, Any]]]:
        """Route each (entity_class, measure) pair and execute.  No LLM call here."""
        entity_classes: list[str] = need.get("entity_classes", []) or ["user_status"]
        measures: list[str] = need.get("measures", [])
        filters: dict[str, Any] = need.get("filters", {}) or {}

        # Build (entity_class, measure) pairs; one pair per measure, or one bare-class entry
        pairs: list[tuple[str, str | None]] = []
        assigned_measures: set[str] = set()
        for ec_name in entity_classes:
            ec_measures = [m for m in measures if MEASURE_TO_ENTITY.get(m) == ec_name]
            if ec_measures:
                for m in ec_measures:
                    pairs.append((ec_name, m))
                    assigned_measures.add(m)
            else:
                pairs.append((ec_name, None))
        # Unassigned measures (entity class not resolved) → attempt anyway
        for m in measures:
            if m not in assigned_measures and m in MEASURE_TO_ENTITY:
                pairs.append((MEASURE_TO_ENTITY[m], m))

        facts_parts: list[str] = []
        retrieval_trace: list[dict[str, Any]] = []
        for ec_name, measure in pairs:
            decision = decide(
                ec_name,
                measure,
                T,
                filters,
                self._registry,
                no_freshness_tiers=self.no_freshness_tiers,
                no_source_routing=self.no_source_routing,
            )
            fact_text, rt = self._execute_one(decision, filters, question)
            facts_parts.append(fact_text)
            retrieval_trace.append(rt)

        return "\n\n".join(facts_parts), retrieval_trace

    def _execute_one(
        self, decision: RouteDecision, filters: dict[str, Any], question: str = ""
    ) -> tuple[str, dict[str, Any]]:
        t0 = time.perf_counter()
        result_text = ""
        llm_inp = llm_out = 0
        extra_trace: dict[str, Any] = {}

        try:
            if decision.route == "staged_sql":
                result_text, llm_inp, llm_out = self._run_staged_sql(decision, filters)
            elif decision.route == "warehouse_live":
                result_text, llm_inp, llm_out = self._run_warehouse_live(decision, filters)
            elif decision.route == "origin_live_saas":
                result_text = self._run_saas_live(decision, filters)
            elif decision.route == "origin_live_mongo":
                result_text = self._run_mongo_live(decision, filters)
            elif decision.route == "docs_index":
                result_text = self._run_docs_index(decision, filters, question)
            elif decision.route == "federated":
                result_text, llm_inp, llm_out = self._run_federated(decision, filters, extra_trace)
            else:
                result_text = f"(unknown route: {decision.route})"
        except Exception as exc:
            result_text = f"(retrieval error: {exc})"

        ms = round((time.perf_counter() - t0) * 1_000, 1)
        staged = decision.route in ("staged_sql", "docs_index")
        trace_entry: dict[str, Any] = {
            "role": "retrieval",
            "entity_class": decision.entity_class,
            "measure": decision.measure,
            "route": decision.route,
            "query_method": decision.query_method,
            "last_refresh": decision.last_refresh.isoformat() if decision.last_refresh else None,
            "cache_miss_reason": decision.cache_miss_reason,
            "staged_vs_live": "staged" if staged else "live",
            "duration_ms": ms,
            "result_preview": result_text[:120],
            "llm_input_tokens": llm_inp,
            "llm_output_tokens": llm_out,
        }
        trace_entry.update(extra_trace)
        return result_text, trace_entry

    def _run_staged_sql(
        self, decision: RouteDecision, filters: dict[str, Any]
    ) -> tuple[str, int, int]:
        """Execute staged SQLite.  Returns (text, inp_tokens, out_tokens)."""
        sql = decision.sql_template
        llm_inp = llm_out = 0

        # If the template requires a param that resolved to None (e.g. cost_center=None
        # means "across all cost centers"), the template WHERE clause is too narrow.
        # Fall through to LLM-generated SQL so it aggregates without that filter.
        if sql is not None and any(filters.get(k) is None for k in decision.sql_params):
            sql = None

        if sql is None and decision.measure is None:
            # Bare entity lookup (no specific measure) — use a generic sample query
            # without calling the LLM.  This keeps the no_semantic_model ablation's
            # retrieval cost at 0 LLM tokens.
            ec = self._registry.get(decision.entity_class)
            if ec and ec.staged_table:
                sql = f"SELECT * FROM {ec.staged_table} LIMIT 5"
            else:
                return f"[staged:{decision.entity_class}] (no staged table)", 0, 0

        if sql is None:
            # Long-tail measure with no template: LLM generates the SQL
            schema_hint = (
                "Tables: staged_users(user_id, cost_center_id, active, hired_at), "
                "staged_assignments(assignment_id, license_id, user_id, product_id), "
                "staged_license_purchases(purchase_id, product_id, cost_center_id, "
                "seats, unit_price_usd, valid_from, valid_until), "
                "staged_cost_centers(cost_center_id, cost_center, business_unit), "
                "staged_vendors(vendor_id, vendor_name, vendor_tier)."
            )
            resp = self._lm.invoke(
                [
                    SystemMessage(
                        content=f"Write a SQLite SELECT query for measure '{decision.measure}'. "
                        f"{schema_hint} Output only the SQL."
                    ),
                    HumanMessage(content=f"Filters: {json.dumps(filters)}"),
                ]
            )
            meta = getattr(resp, "usage_metadata", {}) or {}
            llm_inp = int(meta.get("input_tokens", 0))
            llm_out = int(meta.get("output_tokens", 0))
            sql = _strip_sql_fences(_extract_text(resp.content))

        params = {k: filters[k] for k in decision.sql_params if filters.get(k) is not None}
        try:
            with self._staged_engine.connect() as conn:
                res = conn.execute(text(sql), params)
                rows = res.fetchmany(50)
                cols = list(res.keys())
            if rows:
                header = " | ".join(str(c) for c in cols)
                body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
                return f"[staged:{decision.entity_class}] {header}\n{body}", llm_inp, llm_out
            return f"[staged:{decision.entity_class}] (0 rows)", llm_inp, llm_out
        except Exception as exc:
            return f"[staged:{decision.entity_class}] (SQL error: {exc})", llm_inp, llm_out

    def _run_warehouse_live(
        self, decision: RouteDecision, filters: dict[str, Any]
    ) -> tuple[str, int, int]:
        """Postgres passthrough.  Templated if known measure, else LLM generates SQL."""
        if self._pg_engine is None:
            return "[live:postgres] (no connection)", 0, 0

        sql = decision.sql_template
        llm_inp = llm_out = 0

        # If the template requires a param that resolved to None, fall through to LLM SQL.
        if sql is not None and any(filters.get(k) is None for k in decision.sql_params):
            sql = None

        if sql is None:
            resp = self._lm.invoke(
                [
                    SystemMessage(content=SQL_WORKER_PROMPT),
                    HumanMessage(
                        content=f"Measure: {decision.measure}. Filters: {json.dumps(filters)}"
                    ),
                ]
            )
            meta = getattr(resp, "usage_metadata", {}) or {}
            llm_inp = int(meta.get("input_tokens", 0))
            llm_out = int(meta.get("output_tokens", 0))
            sql = _strip_sql_fences(_extract_text(resp.content))

        params = {k: filters[k] for k in decision.sql_params if filters.get(k) is not None}
        try:
            with self._pg_engine.connect() as conn:
                res = conn.execute(text(sql), params)
                rows = res.fetchmany(20)
                cols = list(res.keys())
            if rows:
                header = " | ".join(str(c) for c in cols)
                body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
                return f"[live:postgres] {header}\n{body}", llm_inp, llm_out
            return "[live:postgres] (0 rows)", llm_inp, llm_out
        except Exception as exc:
            return f"[live:postgres] (error: {exc})", llm_inp, llm_out

    def _run_saas_live(self, decision: RouteDecision, filters: dict[str, Any]) -> str:
        if self._saas is None:
            return "[live:saas] (no connection)"
        try:
            measure = decision.measure or ""
            if "utilization" in measure:
                sku = filters.get("product_sku", "")
                if sku:
                    r = self._saas.raw_get(f"/products/{sku}/current-utilization")
                    return f"[live:saas] {json.dumps(r, default=str)[:300]}"
            elif "ticket" in measure:
                r = self._saas.raw_get("/tickets")
                return f"[live:saas] {json.dumps(r, default=str)[:300]}"
        except Exception as exc:
            return f"[live:saas] (error: {exc})"
        return "[live:saas] (no applicable endpoint)"

    def _run_mongo_live(self, decision: RouteDecision, filters: dict[str, Any]) -> str:
        if self._mongo_db is None:
            return "[live:mongo] (no connection)"
        coll_map = {
            "user_status": "users",
            "assignments": "assignments",
            "cost_center_membership": "users",
        }
        coll = coll_map.get(decision.entity_class, decision.entity_class)
        try:
            query: dict[str, Any] = {}
            if filters.get("cost_center"):
                query["cost_center_id"] = filters["cost_center"]
            docs = list(self._mongo_db[coll].find(query, {"_id": 0}).limit(20))
            return f"[live:mongo:{coll}] {json.dumps(docs, default=str)[:300]}"
        except Exception as exc:
            return f"[live:mongo] (error: {exc})"

    def _run_federated(
        self,
        decision: RouteDecision,
        filters: dict[str, Any],
        extra_trace: dict[str, Any],
    ) -> tuple[str, int, int]:
        """Execute staged SQLite + live Postgres and join in Python.

        Fills extra_trace with staged_row_count and live_row_count so the
        caller can include them in the retrieval trace entry.
        """
        if decision.measure is None or decision.measure not in FEDERATED_TEMPLATES:
            return f"[federated:{decision.entity_class}] (no template)", 0, 0

        tmpl = FEDERATED_TEMPLATES[decision.measure]
        staged_params = {k: filters[k] for k in tmpl.staged_params if filters.get(k) is not None}
        live_params = {k: filters[k] for k in tmpl.warehouse_params if filters.get(k) is not None}

        # Guard: if any required staged param is absent, the SQL would raise
        # InvalidRequestError on the unbound :param placeholder.  Return an
        # informative message instead of crashing the whole task.
        missing = [k for k in tmpl.staged_params if filters.get(k) is None]
        if missing:
            return (
                f"[federated:{decision.entity_class}] "
                f"(skipped: missing params {missing} for measure {decision.measure!r}; "
                "check need-resolution measure selection)",
                0,
                0,
            )

        # ── staged SQLite ──────────────────────────────────────────────────────
        staged_rows: list[dict[str, Any]] = []
        try:
            with self._staged_engine.connect() as conn:
                res = conn.execute(text(tmpl.staged_sql), staged_params)
                cols = list(res.keys())
                staged_rows = [dict(zip(cols, row)) for row in res.fetchall()]
        except Exception as exc:
            return f"[federated:{decision.entity_class}] (staged error: {exc})", 0, 0

        # ── live Postgres ─────────────────────────────────────────────────────
        live_rows: list[dict[str, Any]] = []
        if self._pg_engine is not None:
            try:
                with self._pg_engine.connect() as conn:
                    res = conn.execute(text(tmpl.warehouse_sql), live_params)
                    cols = list(res.keys())
                    live_rows = [dict(zip(cols, row)) for row in res.fetchall()]
            except Exception as exc:
                return f"[federated:{decision.entity_class}] (live error: {exc})", 0, 0

        extra_trace["staged_row_count"] = len(staged_rows)
        extra_trace["live_row_count"] = len(live_rows)

        # ── Python join ───────────────────────────────────────────────────────
        try:
            result = tmpl.join_fn(staged_rows, live_rows)
        except Exception as exc:
            return f"[federated:{decision.entity_class}] (join error: {exc})", 0, 0

        # Mirror the staged-SQL format (measure + value) so synthesis LLM
        # sees a labelled scalar and does not re-derive from raw counts.
        measure_label = decision.measure or "value"
        return (
            f"[federated:{decision.entity_class}] {measure_label}\n{result}",
            0,
            0,
        )

    def _run_docs_index(
        self,
        decision: RouteDecision,
        filters: dict[str, Any] | None = None,
        question: str = "",
    ) -> str:
        coll = self._full_coll if self.no_source_routing else self._docs_coll
        if coll is None:
            return "[docs_index] (no collection)"
        try:
            # Use the original question as the semantic query when available;
            # fall back to the measure name so the caller always gets something.
            query_text = question or decision.measure or decision.entity_class
            q_emb: list[list[float]] = self._embed.encode(
                [query_text], show_progress_bar=False
            ).tolist()
            top_k = _FULL_INDEX_TOP_K if self.no_source_routing else _DOCS_TOP_K
            n = min(top_k, coll.count())
            if n == 0:
                return "[docs_index] (empty index)"

            # When contract_id is known and source routing is enabled, restrict the
            # ChromaDB query to the single matching document so retrieval is exact.
            query_kwargs: dict[str, Any] = {
                "query_embeddings": q_emb,
                "n_results": n,
                "include": ["documents"],
            }
            contract_id = (filters or {}).get("contract_id")
            if contract_id and not self.no_source_routing:
                query_kwargs["where"] = {"source": f"{contract_id}.md"}
                query_kwargs["n_results"] = min(n, max(1, coll.count()))

            results = coll.query(**query_kwargs)
            docs: list[str] = results["documents"][0] if results["documents"] else []
            return "[docs_index] " + " || ".join(docs[:3])
        except Exception as exc:
            return f"[docs_index] (error: {exc})"

    # ── LLM call 2: synthesis ─────────────────────────────────────────────────

    def _synthesize(self, question: str, facts_text: str, answer_type: str) -> tuple[str, int, int]:
        sys_p = system_prompt(answer_type)
        human = (
            f"Pre-computed facts (these are the final retrieved values — "
            f"do not re-derive or second-guess them):\n{facts_text}\n\nQuestion: {question}"
        )
        response = self._lm.invoke([SystemMessage(content=sys_p), HumanMessage(content=human)])
        meta = getattr(response, "usage_metadata", {}) or {}
        inp = int(meta.get("input_tokens", 0))
        out = int(meta.get("output_tokens", 0))
        return _extract_text(response.content).strip(), inp, out

    # ── answer() ──────────────────────────────────────────────────────────────

    def answer(self, task: Task) -> ArmResult:
        if self._lm is None:
            raise RuntimeError("GroundingArm.setup() must be called before answer()")

        t0 = time.perf_counter()
        trace: list[dict[str, Any]] = []
        total_inp = total_out = 0

        # ── LLM call 1: need-resolution ───────────────────────────────────────
        need, nr_inp, nr_out = self._resolve_needs(task.question_text)
        total_inp += nr_inp
        total_out += nr_out

        # Inject derived filters that the LLM cannot reliably extract from question text:
        #   cutoff_date — from window_days / idle_days relative to T
        #   threshold   — numeric spend threshold for SV6 tasks
        #   cost_center — direct cc param name differs from the filter schema key
        filters: dict[str, Any] = need.get("filters") or {}
        window: int | None = task.params.get("window_days") or task.params.get("idle_days")
        if window is not None:
            filters["cutoff_date"] = (task.T - timedelta(days=int(window))).isoformat()
        if "threshold" in task.params:
            filters["threshold"] = float(task.params["threshold"])
        if "cc" in task.params and not filters.get("cost_center"):
            filters["cost_center"] = str(task.params["cc"])
        if "contract_id" in task.params and not filters.get("contract_id"):
            filters["contract_id"] = str(task.params["contract_id"])
        # Normalize product identifier: the LLM may fill either product_sku or product_id
        # (both are string SKUs like "prd_0014"). Federated staged SQL params use product_id;
        # copy across so the missing-param guard never fires due to naming variance alone.
        if filters.get("product_sku") and not filters.get("product_id"):
            filters["product_id"] = filters["product_sku"]
        elif filters.get("product_id") and not filters.get("product_sku"):
            filters["product_sku"] = filters["product_id"]
        need["filters"] = filters

        # Measure disambiguation: if a cost_center filter is present but product_id
        # is absent, product-scoped measures like zero_usage_license_count can't execute
        # (their federated staged SQL requires :product_id).  Remap to the cost-center
        # variant where one exists so the correct federated template runs instead.
        _CC_MEASURE_REMAP: dict[str, str] = {
            "zero_usage_license_count": "idle_license_count_cc",
        }
        if filters.get("cost_center") and not filters.get("product_id"):
            remapped = [_CC_MEASURE_REMAP.get(m, m) for m in need.get("measures", [])]
            if remapped != need.get("measures", []):
                need["measures"] = remapped
                # Re-infer entity classes from the remapped measures
                need["entity_classes"] = list(
                    {MEASURE_TO_ENTITY[m] for m in remapped if m in MEASURE_TO_ENTITY}
                )

        trace.append(
            {
                "role": "need_resolution",
                "entity_classes": need.get("entity_classes", []),
                "measures": need.get("measures", []),
                "filters": need.get("filters", {}),
                "input_tokens": nr_inp,
                "output_tokens": nr_out,
            }
        )

        # ── Deterministic routing + query execution ───────────────────────────
        facts_text, retrieval_trace = self._execute_retrieval(need, task.T, task.question_text)
        for rt in retrieval_trace:
            total_inp += rt.get("llm_input_tokens", 0)
            total_out += rt.get("llm_output_tokens", 0)
        trace.extend(retrieval_trace)

        # ── LLM call 2: synthesis ─────────────────────────────────────────────
        answer_raw, syn_inp, syn_out = self._synthesize(
            task.question_text, facts_text, task.answer_type
        )
        total_inp += syn_inp
        total_out += syn_out
        trace.append(
            {
                "role": "synthesis",
                "input_tokens": syn_inp,
                "output_tokens": syn_out,
                "answer": answer_raw[:100],
            }
        )

        latency_s = round(time.perf_counter() - t0, 3)
        parsed = parse(answer_raw, task.answer_type)
        c = cost_usd(self._model, total_inp, total_out, self._embedding_tokens)

        return ArmResult(
            task_id=task.task_id,
            answer_raw=answer_raw,
            answer_parsed=parsed.value,
            input_tokens=total_inp,
            output_tokens=total_out,
            embedding_tokens=self._embedding_tokens,
            cost_usd=c,
            latency_s=latency_s,
            tool_calls=[],
            trace=trace,
        )
