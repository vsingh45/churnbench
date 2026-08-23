"""Arm 2 — Classic RAG ('flatten-everything' anti-pattern).

setup(config, T_prime):
  Serialize ALL fabric sources as of T_prime into text chunks.
  Embed chunks with a local sentence-transformers model (all-MiniLM-L6-v2).
  Store in an in-memory ChromaDB collection.
  Record T_prime in index metadata — the index is legitimately stale after
  the harness re-projects at T.

answer(task):
  Embed the question, retrieve top-k=8 chunks, one LLM call with chunks + question.
  No tools, no iteration — a single retrieval + generation step.

Anti-pattern faithfully executed:
  - A practitioner dumps all their data into a vector index and retrieves on
    every question.  Simple to build, genuinely works for prose-heavy (tier-3)
    questions, struggles with aggregate arithmetic.
  - Chunking strategy is sensible (not pathological): rows serialized as natural-
    language sentences, each row is one chunk, docs split by section.
  - Retrieval uses the question verbatim (no query rewriting).
  - Embedding cost is $0 (local model), but token counts are reported.
  - The index is built once at T_prime and never refreshed — staleness is the
    phenomenon under study.

This arm should win on tier-3 contract prose questions and lose on tier-1/2
aggregate questions where the LLM must reason over retrieved fragments.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Any

import chromadb

from churnbench.arms.base import ArmResult, BaseArm, FabricConfig, active_model, cost_usd, llm
from churnbench.arms.prompts import system_prompt
from churnbench.eval.parsing import parse
from churnbench.fabric.saas_client import SaasClient
from churnbench.tasks.schema import Task

_TOP_K = 8
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def _rough_token_count(texts: list[str]) -> int:
    """Estimate embedding token count via word-count × 1.3 heuristic."""
    return int(sum(len(t.split()) for t in texts) * 1.3)


def _chunks_postgres(pg_url: str) -> list[dict[str, Any]]:
    """Serialize Postgres rows as natural-language sentences."""
    from sqlalchemy import create_engine, text as _text

    chunks: list[dict[str, Any]] = []
    try:
        engine = create_engine(pg_url, future=True)
        with engine.connect() as conn:
            # Vendors
            for row in conn.execute(
                _text("SELECT vendor_id, vendor_name, vendor_tier FROM sam.dim_vendor")
            ).mappings():
                chunks.append(
                    {
                        "text": (
                            f"[dim_vendor vendor_id={row['vendor_id']}] "
                            f"Vendor '{row['vendor_name']}' is classified as {row['vendor_tier']} tier."
                        ),
                        "meta": {"source": "postgres", "table": "dim_vendor"},
                    }
                )

            # Cost centers
            for row in conn.execute(
                _text("SELECT cost_center_id, cost_center, business_unit FROM sam.dim_cost_center")
            ).mappings():
                chunks.append(
                    {
                        "text": (
                            f"[dim_cost_center id={row['cost_center_id']}] "
                            f"Cost center '{row['cost_center']}' belongs to business unit "
                            f"'{row['business_unit']}'."
                        ),
                        "meta": {"source": "postgres", "table": "dim_cost_center"},
                    }
                )

            # Products
            for row in conn.execute(
                _text(
                    "SELECT product_id, product_sku, product_name, license_model "
                    "FROM sam.dim_product"
                )
            ).mappings():
                chunks.append(
                    {
                        "text": (
                            f"[dim_product sku={row['product_sku']}] "
                            f"Product '{row['product_name']}' (sku={row['product_sku']}) "
                            f"uses {row['license_model']} licensing."
                        ),
                        "meta": {"source": "postgres", "table": "dim_product"},
                    }
                )

            # License purchases
            for row in conn.execute(
                _text(
                    "SELECT purchase_id, product_id, cost_center_id, purchase_date, "
                    "seats, unit_price_usd, contract_id, valid_from, valid_until "
                    "FROM sam.fact_license_purchase"
                )
            ).mappings():
                chunks.append(
                    {
                        "text": (
                            f"[fact_license_purchase id={row['purchase_id']}] "
                            f"Product {row['product_id']} purchased for cost center "
                            f"{row['cost_center_id']}: {row['seats']} seat(s) at "
                            f"${row['unit_price_usd']}/seat/month, contract {row['contract_id']}, "
                            f"valid {row['valid_from']} to {row['valid_until']}."
                        ),
                        "meta": {"source": "postgres", "table": "fact_license_purchase"},
                    }
                )

            # Consumption — aggregate by product+user to keep chunks dense
            for row in conn.execute(
                _text(
                    "SELECT product_id, user_ext_id, "
                    "SUM(session_minutes) AS mins, SUM(api_calls) AS calls "
                    "FROM sam.fact_consumption_event "
                    "GROUP BY product_id, user_ext_id"
                )
            ).mappings():
                chunks.append(
                    {
                        "text": (
                            f"[fact_consumption product={row['product_id']} "
                            f"user={row['user_ext_id']}] "
                            f"Total usage: {row['mins']} session minutes, "
                            f"{row['calls']} API calls."
                        ),
                        "meta": {"source": "postgres", "table": "fact_consumption_event"},
                    }
                )
        engine.dispose()
    except Exception:
        pass  # DB not available (test / no docker-compose)
    return chunks


def _chunks_mongo(mongo_url: str) -> list[dict[str, Any]]:
    """Serialize MongoDB operational collections as text chunks."""
    from pymongo import MongoClient

    chunks: list[dict[str, Any]] = []
    try:
        client: MongoClient[Any] = MongoClient(mongo_url)
        db = client["sam_ops"]

        for coll_name in [
            "users",
            "active_licenses",
            "assignments",
            "entitlements",
            "tickets",
            "utilization_current",
        ]:
            for doc in db[coll_name].find({}, {"_id": 0}).limit(500):
                chunks.append(
                    {
                        "text": (f"[mongo:{coll_name}] " + json.dumps(doc, default=str)),
                        "meta": {"source": "mongo", "collection": coll_name},
                    }
                )
        client.close()
    except Exception:
        pass
    return chunks


def _chunks_saas(saas: SaasClient, mongo_url: str) -> list[dict[str, Any]]:
    """Dump SaaS tickets and per-product utilization as chunks.

    Discovers product SKUs via MongoDB to avoid calling every possible path.
    This is one call per product + one ticket call — O(products), not O(users),
    so it stays well within the 60 req/min rate limit for typical ledger sizes.
    """
    chunks: list[dict[str, Any]] = []
    try:
        # Tickets (one call)
        tickets = saas.get_tickets()
        for ticket in tickets:
            chunks.append(
                {
                    "text": "[saas:ticket] " + json.dumps(ticket, default=str),
                    "meta": {"source": "saas", "endpoint": "/tickets"},
                }
            )

        # Per-product utilization — discover SKUs from MongoDB
        try:
            from pymongo import MongoClient

            mc: MongoClient[Any] = MongoClient(mongo_url)
            skus = [
                doc["product_sku"]
                for doc in mc["sam_ops"]["utilization_current"].find(
                    {}, {"product_sku": 1, "_id": 0}
                )
            ]
            mc.close()
        except Exception:
            skus = []

        for sku in skus:
            try:
                util = saas.get_product_utilization(sku)
                chunks.append(
                    {
                        "text": f"[saas:utilization sku={sku}] " + json.dumps(util, default=str),
                        "meta": {
                            "source": "saas",
                            "endpoint": f"/products/{sku}/current-utilization",
                        },
                    }
                )
            except Exception:
                pass
    except Exception:
        pass
    return chunks


def _chunks_docs(docs_dir: Path) -> list[dict[str, Any]]:
    """Split contract Markdown files by section (## headers)."""
    chunks: list[dict[str, Any]] = []
    for fpath in sorted(docs_dir.glob("*.md")):
        doc_id = fpath.stem
        try:
            content = fpath.read_text()
        except OSError:
            continue
        # Split by section headers; keep provenance prefix on each chunk
        sections = [s.strip() for s in content.split("\n## ") if s.strip()]
        for i, section in enumerate(sections):
            chunks.append(
                {
                    "text": f"[docs:{doc_id} section={i}] {section[:1000]}",
                    "meta": {"source": "docs", "doc_id": doc_id, "section": i},
                }
            )
    return chunks


class ClassicRagArm(BaseArm):
    """Arm 2: classic RAG — serialize everything, retrieve, generate once.

    The index is built once during setup() at T_prime.  After the harness
    re-projects the fabric at T and begins asking questions, the index is
    legitimately stale — that's the phenomenon under study.
    """

    def __init__(
        self,
        *,
        _saas_client: SaasClient | None = None,
        _embed_model: Any = None,  # SentenceTransformer | FakeEmbedModel
    ) -> None:
        """
        _saas_client and _embed_model are injection points for testing.
        Production code leaves both as None (defaults applied in setup).
        """
        self._saas_override = _saas_client
        self._embed_override = _embed_model
        self._collection: Any = None  # chromadb.Collection
        self._embed: Any = None
        self._embedding_tokens: int = 0
        self._model: str = active_model()
        self._mongo_url: str = ""

    def setup(self, config: FabricConfig, T_prime: date) -> None:
        """Serialize all fabric layers at T_prime and build the vector index.

        The fabric must already be projected at T_prime (done by the harness).
        This method only reads — it does not project.
        """
        self._model = active_model()
        self._mongo_url = config.mongo_url

        # Load embedding model
        if self._embed_override is not None:
            self._embed = self._embed_override
        else:
            from sentence_transformers import SentenceTransformer

            self._embed = SentenceTransformer(_EMBED_MODEL_NAME)

        # Create in-memory ChromaDB collection
        client = chromadb.EphemeralClient()
        self._collection = client.create_collection(
            name="churnbench",
            metadata={"T_prime": T_prime.isoformat(), "hnsw:space": "cosine"},
        )

        # Collect chunks from all four sources
        saas = self._saas_override or SaasClient(config.saas_base_url)
        chunks: list[dict[str, Any]] = []
        chunks += _chunks_postgres(config.pg_url)
        chunks += _chunks_mongo(config.mongo_url)
        chunks += _chunks_saas(saas, config.mongo_url)
        chunks += _chunks_docs(config.docs_dir)
        if saas is not self._saas_override:
            saas.close()

        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        metadatas = [c["meta"] for c in chunks]
        self._embedding_tokens = _rough_token_count(texts)

        embeddings: list[list[float]] = self._embed.encode(
            texts, show_progress_bar=False, batch_size=64
        ).tolist()

        self._collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=[f"c{i}" for i in range(len(chunks))],
            metadatas=metadatas,
        )

    def answer(self, task: Task) -> ArmResult:
        if self._collection is None or self._embed is None:
            raise RuntimeError("ClassicRagArm.setup() must be called before answer()")

        t0 = time.perf_counter()

        # Retrieve top-k chunks with the question verbatim
        q_emb: list[list[float]] = self._embed.encode(
            [task.question_text], show_progress_bar=False
        ).tolist()
        results = self._collection.query(
            query_embeddings=q_emb,
            n_results=min(_TOP_K, self._collection.count()),
            include=["documents"],
        )
        docs: list[str] = results["documents"][0] if results["documents"] else []

        # Single LLM call — no tools, no iteration
        lm = llm(self._model)
        sys = system_prompt(task.answer_type)
        context_block = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))
        human_msg = f"Relevant data:\n{context_block}\n\n" f"Question: {task.question_text}"

        from langchain_core.messages import HumanMessage, SystemMessage

        response = lm.invoke([SystemMessage(content=sys), HumanMessage(content=human_msg)])

        latency_s = time.perf_counter() - t0
        answer_raw = ""
        if isinstance(response.content, str):
            answer_raw = response.content.strip()
        elif isinstance(response.content, list):
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    answer_raw = str(block.get("text", "")).strip()
                    break

        meta = getattr(response, "usage_metadata", {}) or {}
        inp = int(meta.get("input_tokens", 0))
        out = int(meta.get("output_tokens", 0))

        parsed = parse(answer_raw, task.answer_type)
        c = cost_usd(self._model, inp, out, self._embedding_tokens)

        trace = [{"role": "context", "content": doc[:200]} for doc in docs] + [
            {"role": "answer", "content": answer_raw}
        ]

        return ArmResult(
            task_id=task.task_id,
            answer_raw=answer_raw,
            answer_parsed=parsed.value,
            input_tokens=inp,
            output_tokens=out,
            embedding_tokens=self._embedding_tokens,
            cost_usd=c,
            latency_s=round(latency_s, 3),
            tool_calls=[],
            trace=trace,
        )

    def teardown(self) -> None:
        # ChromaDB EphemeralClient is in-memory; GC handles it.
        self._collection = None
        self._embed = None
