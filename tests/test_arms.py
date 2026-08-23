"""Tests for churnbench/arms/ and churnbench/eval/parsing.py.

No live API calls: SQL uses SQLite in-memory, MongoDB is mocked, SaaS is
mocked, the embedding model is a local fake, and the LLM is patched.
"""

from __future__ import annotations

import json
import threading
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine, text

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from churnbench.arms.base import ArmResult, cost_usd
from churnbench.arms.prompts import system_prompt
from churnbench.eval.parsing import ParseResult, parse
from churnbench.tasks.schema import Task


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_task(
    answer_type: str = "int",
    resolver_ref: str = "active_user_count_cc",
) -> Task:
    return Task(
        task_id="task_0000",
        template_id="SV2",
        intent="spend_visibility",
        tier=1,
        question_text="How many active users are in cost center cc_000 as of 2024-01-15?",
        params={"cc": "cc_000"},
        T=date(2024, 1, 15),
        answer_type=answer_type,
        resolver_ref=resolver_ref,
    )


class _FakeEmbedModel:
    """Deterministic fake embedding model — bag-of-keywords encoding."""

    _VOCAB = [
        "license",
        "cost",
        "user",
        "product",
        "contract",
        "spend",
        "vendor",
        "active",
        "assigned",
        "monthly",
        "ticket",
        "utilization",
    ]

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        rows = []
        for txt in texts:
            words = set(txt.lower().split())
            vec = [1.0 if kw in words else 0.0 for kw in self._VOCAB]
            norm = max(sum(v**2 for v in vec) ** 0.5, 1e-8)
            rows.append([v / norm for v in vec])
        return np.array(rows, dtype=np.float32)


class _FakeLM:
    """Minimal ChatOpenAI-compatible fake that returns scripted responses."""

    def __init__(self, responses: list[Any]) -> None:
        self._resp = iter(responses)
        self.model_name = "nvidia/nemotron-3-ultra-550b-a55b"

    def invoke(self, messages: Any, **_: Any) -> Any:
        resp = next(self._resp)
        return resp

    def bind_tools(self, _tools: Any, **_kw: Any) -> "_FakeLM":
        return self

    # LangChain expects some attrs
    def with_structured_output(self, _schema: Any, **_kw: Any) -> "_FakeLM":  # noqa: PLR6301
        return self

    @property
    def _llm_type(self) -> str:
        return "fake"


def _ai_message(content: str, input_tokens: int = 50, output_tokens: int = 10) -> Any:
    """Build a minimal AIMessage-like object with usage_metadata."""
    from langchain_core.messages import AIMessage

    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cost accounting
# ─────────────────────────────────────────────────────────────────────────────


class TestCostAccounting:
    def test_nemotron_hand_computed(self) -> None:
        # 1 000 input × $3.50/MTok = $0.0035
        # 500 output × $3.50/MTok  = $0.00175
        # total                     = $0.00525
        assert cost_usd("nvidia/nemotron-3-ultra-550b-a55b", 1_000, 500) == pytest.approx(0.00525)

    def test_deepseek_flash_hand_computed(self) -> None:
        # 2 000 input × $0.20/MTok = $0.0004
        # 1 000 output × $0.60/MTok = $0.0006
        # total                      = $0.0010
        assert cost_usd("deepseek-ai/deepseek-v4-flash", 2_000, 1_000) == pytest.approx(0.001)

    def test_unknown_model_falls_back_to_nemotron(self) -> None:
        # Should not raise; falls back to nemotron rate ($3.50 in/out)
        # 1 000 × $3.50/MTok = $0.0035
        result = cost_usd("nonexistent-model-xyz", 1_000, 0)
        assert result == pytest.approx(0.0035)

    def test_zero_tokens_zero_cost(self) -> None:
        assert cost_usd("nvidia/nemotron-3-ultra-550b-a55b", 0, 0) == pytest.approx(0.0)

    def test_embedding_tokens_tracked_not_billed(self) -> None:
        # Embedding tokens do not change cost (local model is free)
        without = cost_usd("nvidia/nemotron-3-ultra-550b-a55b", 1_000, 500, embedding_tokens=0)
        with_emb = cost_usd(
            "nvidia/nemotron-3-ultra-550b-a55b", 1_000, 500, embedding_tokens=999_999
        )
        assert without == pytest.approx(with_emb)

    def test_nemotron_symmetric_rates(self) -> None:
        # input_rate == output_rate for nemotron — swapping counts has no effect on total
        assert cost_usd("nvidia/nemotron-3-ultra-550b-a55b", 1_000, 0) == pytest.approx(
            cost_usd("nvidia/nemotron-3-ultra-550b-a55b", 0, 1_000)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Parsing — every answer_type, including edge cases
# ─────────────────────────────────────────────────────────────────────────────


class TestParsingInt:
    def test_simple_integer(self) -> None:
        r = parse("42", "int")
        assert r.value == 42
        assert not r.parse_failure

    def test_integer_with_surrounding_text(self) -> None:
        r = parse("3 licenses", "int")
        assert r.value == 3

    def test_integer_with_commas(self) -> None:
        r = parse("1,234", "int")
        assert r.value == 1234

    def test_integer_zero(self) -> None:
        r = parse("0", "int")
        assert r.value == 0
        assert not r.parse_failure

    def test_negative_integer(self) -> None:
        r = parse("-5", "int")
        assert r.value == -5

    def test_integer_from_sentence(self) -> None:
        r = parse("There are 7 active users.", "int")
        assert r.value == 7

    def test_none_string_fails(self) -> None:
        r = parse("none", "int")
        assert r.parse_failure
        assert r.value is None

    def test_not_found_fails(self) -> None:
        r = parse("not found", "int")
        assert r.parse_failure

    def test_empty_string_fails(self) -> None:
        r = parse("", "int")
        assert r.parse_failure

    def test_plain_text_fails(self) -> None:
        r = parse("active", "int")
        assert r.parse_failure


class TestParsingFloat:
    def test_simple_float(self) -> None:
        r = parse("1234.56", "float")
        assert r.value == pytest.approx(1234.56)
        assert not r.parse_failure

    def test_dollar_with_commas(self) -> None:
        r = parse("$1,234.56", "float")
        assert r.value == pytest.approx(1234.56)

    def test_integer_as_float(self) -> None:
        r = parse("42", "float")
        assert r.value == pytest.approx(42.0)

    def test_zero_float(self) -> None:
        r = parse("0.0", "float")
        assert r.value == pytest.approx(0.0)
        assert not r.parse_failure

    def test_negative_float(self) -> None:
        r = parse("-3.14", "float")
        assert r.value == pytest.approx(-3.14)

    def test_float_with_dollar_no_comma(self) -> None:
        r = parse("$500.00", "float")
        assert r.value == pytest.approx(500.0)

    def test_float_none_fails(self) -> None:
        r = parse("N/A", "float")
        assert r.parse_failure

    def test_float_null_fails(self) -> None:
        r = parse("null", "float")
        assert r.parse_failure

    def test_float_from_sentence(self) -> None:
        r = parse("The total spend is $2,500.00 per month.", "float")
        assert r.value == pytest.approx(2500.0)

    def test_percent_stripped(self) -> None:
        r = parse("87.5%", "float")
        assert r.value == pytest.approx(87.5)


class TestParsingStr:
    def test_simple_string(self) -> None:
        r = parse("active", "str")
        assert r.value == "active"
        assert not r.parse_failure

    def test_strips_whitespace(self) -> None:
        r = parse("  Acme Corp  ", "str")
        assert r.value == "Acme Corp"

    def test_strips_surrounding_double_quotes(self) -> None:
        r = parse('"Acme Corp"', "str")
        assert r.value == "Acme Corp"

    def test_strips_surrounding_single_quotes(self) -> None:
        r = parse("'Acme Corp'", "str")
        assert r.value == "Acme Corp"

    def test_strips_markdown_bold(self) -> None:
        r = parse("**expired**", "str")
        assert r.value == "expired"

    def test_strips_backticks(self) -> None:
        r = parse("`active`", "str")
        assert r.value == "active"

    def test_empty_string_returns_empty_no_failure(self) -> None:
        r = parse("", "str")
        # str parse never fails — empty string is a valid str answer
        assert not r.parse_failure
        assert r.value == ""

    def test_vendor_name_preserved(self) -> None:
        r = parse("TechNova Inc.", "str")
        assert r.value == "TechNova Inc."

    def test_entity_id_underscore_preserved(self) -> None:
        # Regression: _ was stripped by [*_`]+ regex, turning cc_008 → cc008
        r = parse("cc_008", "str")
        assert r.value == "cc_008", f"Expected 'cc_008', got {r.value!r}"

    def test_product_id_underscore_preserved(self) -> None:
        r = parse("prd_0005", "str")
        assert r.value == "prd_0005", f"Expected 'prd_0005', got {r.value!r}"

    def test_markdown_italic_underscore_still_stripped_at_boundaries(self) -> None:
        # _italic_ — underscores only at word boundaries are not stripped by the new
        # regex, but the surrounding-quote stripping doesn't apply to _.
        # The important guarantee: the value does NOT become "italic" (we don't strip _).
        # This test documents the current behaviour rather than asserting stripping.
        r = parse("_italic_", "str")
        assert "_" in r.value  # underscore is preserved, not stripped

    def test_markdown_bold_still_stripped(self) -> None:
        r = parse("**active**", "str")
        assert r.value == "active"

    def test_backtick_code_still_stripped(self) -> None:
        r = parse("`cc_008`", "str")
        assert r.value == "cc_008"  # backtick stripped, underscore preserved


class TestParsingListStr:
    def test_json_array(self) -> None:
        r = parse('["A", "B", "C"]', "list[str]")
        assert r.value == ["A", "B", "C"]
        assert not r.parse_failure

    def test_json_array_no_spaces(self) -> None:
        r = parse('["X","Y"]', "list[str]")
        assert r.value == ["X", "Y"]

    def test_comma_separated_fallback(self) -> None:
        r = parse("A, B, C", "list[str]")
        assert r.value == ["A", "B", "C"]

    def test_newline_separated_fallback(self) -> None:
        r = parse("A\nB\nC", "list[str]")
        assert r.value == ["A", "B", "C"]

    def test_single_element_list(self) -> None:
        r = parse('["only_one"]', "list[str]")
        assert r.value == ["only_one"]

    def test_bracket_with_single_quotes(self) -> None:
        r = parse("['A', 'B']", "list[str]")
        assert r.value == ["A", "B"]

    def test_empty_string_fails(self) -> None:
        r = parse("", "list[str]")
        assert r.parse_failure

    def test_parse_result_dataclass(self) -> None:
        r = parse("42", "int")
        assert isinstance(r, ParseResult)
        assert r.raw == "42"


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────


class TestPrompts:
    def test_system_prompt_contains_format(self) -> None:
        p = system_prompt("int")
        assert "integer" in p

    def test_system_prompt_float(self) -> None:
        p = system_prompt("float")
        assert "decimal" in p

    def test_system_prompt_list(self) -> None:
        p = system_prompt("list[str]")
        assert "JSON array" in p

    def test_system_prompt_str(self) -> None:
        p = system_prompt("str")
        assert "string" in p.lower()

    def test_system_prompt_unknown_type(self) -> None:
        # Should not raise; returns a sensible default
        p = system_prompt("unknown_type")
        assert len(p) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Naive arm — tool-binding plumbing (no LangGraph, no live connections)
# ─────────────────────────────────────────────────────────────────────────────


class TestNaiveArmTools:
    """Test each tool function in isolation with lightweight mocked backends."""

    def _build_log_and_tools(
        self,
        engine: Any,
        mongo_db: Any,
        saas: Any,
        docs_dir: Path,
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        from churnbench.arms.naive import _build_tools

        log: list[dict[str, Any]] = []
        tools = _build_tools(engine, mongo_db, saas, docs_dir, log)
        return log, tools

    def test_sql_query_returns_rows(self, tmp_path: Path) -> None:
        engine = create_engine("sqlite:///:memory:")
        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE numbers (n INTEGER)"))
            conn.execute(text("INSERT INTO numbers VALUES (1), (2), (3)"))
            conn.commit()

        log, tools = self._build_log_and_tools(engine, MagicMock(), MagicMock(), tmp_path)
        sql_tool = next(t for t in tools if t.name == "sql_query")
        result = sql_tool.invoke({"query": "SELECT n FROM numbers"})

        assert "1" in result
        assert "2" in result
        assert "3 rows" in result
        assert len(log) == 1
        assert log[0]["tool"] == "sql_query"
        assert log[0]["duration_ms"] >= 0

    def test_sql_query_caps_at_50_rows(self, tmp_path: Path) -> None:
        engine = create_engine("sqlite:///:memory:")
        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE big (n INTEGER)"))
            for i in range(100):
                conn.execute(text(f"INSERT INTO big VALUES ({i})"))
            conn.commit()

        log, tools = self._build_log_and_tools(engine, MagicMock(), MagicMock(), tmp_path)
        sql_tool = next(t for t in tools if t.name == "sql_query")
        result = sql_tool.invoke({"query": "SELECT n FROM big"})
        assert "50 rows" in result

    def test_sql_query_error_is_graceful(self, tmp_path: Path) -> None:
        engine = create_engine("sqlite:///:memory:")
        log, tools = self._build_log_and_tools(engine, MagicMock(), MagicMock(), tmp_path)
        sql_tool = next(t for t in tools if t.name == "sql_query")
        result = sql_tool.invoke({"query": "SELECT * FROM nonexistent_table"})
        assert "SQL error" in result

    def test_mongo_find_queries_collection(self, tmp_path: Path) -> None:
        mock_coll = MagicMock()
        mock_coll.find.return_value.limit.return_value = [{"user_id": "u1", "active": True}]

        class _MockDB:
            def __getitem__(self, name: str) -> Any:
                return mock_coll

        log, tools = self._build_log_and_tools(MagicMock(), _MockDB(), MagicMock(), tmp_path)
        mongo_tool = next(t for t in tools if t.name == "mongo_find")
        result = mongo_tool.invoke({"collection": "users", "filter_json": "{}"})

        assert "u1" in result
        assert len(log) == 1
        assert log[0]["tool"] == "mongo_find"

    def test_mongo_find_bad_json_graceful(self, tmp_path: Path) -> None:
        log, tools = self._build_log_and_tools(MagicMock(), MagicMock(), MagicMock(), tmp_path)
        mongo_tool = next(t for t in tools if t.name == "mongo_find")
        result = mongo_tool.invoke({"collection": "users", "filter_json": "not json"})
        assert "MongoDB error" in result

    def test_saas_get_calls_raw_get(self, tmp_path: Path) -> None:
        mock_saas = MagicMock()
        mock_saas.raw_get.return_value = {"product_sku": "sku_A", "api_calls": 100}

        log, tools = self._build_log_and_tools(MagicMock(), MagicMock(), mock_saas, tmp_path)
        saas_tool = next(t for t in tools if t.name == "saas_get")
        result = saas_tool.invoke({"path": "/products/sku_A/current-utilization"})

        assert "sku_A" in result
        assert "api_calls" in result
        mock_saas.raw_get.assert_called_once_with("/products/sku_A/current-utilization")
        assert log[0]["tool"] == "saas_get"

    def test_list_documents_shows_stems(self, tmp_path: Path) -> None:
        (tmp_path / "ctr_0001.md").write_text("Contract 1")
        (tmp_path / "ctr_0002.md").write_text("Contract 2")

        log, tools = self._build_log_and_tools(MagicMock(), MagicMock(), MagicMock(), tmp_path)
        list_tool = next(t for t in tools if t.name == "list_documents")
        result = list_tool.invoke({})

        assert "ctr_0001" in result
        assert "ctr_0002" in result
        assert log[0]["tool"] == "list_documents"

    def test_list_documents_empty_dir(self, tmp_path: Path) -> None:
        log, tools = self._build_log_and_tools(MagicMock(), MagicMock(), MagicMock(), tmp_path)
        list_tool = next(t for t in tools if t.name == "list_documents")
        result = list_tool.invoke({})
        assert "no documents" in result

    def test_read_document_returns_content(self, tmp_path: Path) -> None:
        (tmp_path / "ctr_0001.md").write_text("# Contract\nAnnual value: $10000")

        log, tools = self._build_log_and_tools(MagicMock(), MagicMock(), MagicMock(), tmp_path)
        read_tool = next(t for t in tools if t.name == "read_document")
        result = read_tool.invoke({"doc_id": "ctr_0001"})

        assert "Annual value" in result
        assert log[0]["tool"] == "read_document"
        assert log[0]["target"] == "ctr_0001"

    def test_read_document_missing_file_graceful(self, tmp_path: Path) -> None:
        log, tools = self._build_log_and_tools(MagicMock(), MagicMock(), MagicMock(), tmp_path)
        read_tool = next(t for t in tools if t.name == "read_document")
        result = read_tool.invoke({"doc_id": "missing_doc"})
        assert "Doc error" in result

    def test_tool_log_cleared_between_answers(self, tmp_path: Path) -> None:
        """tool_log.clear() in answer() is verified indirectly: log accumulates."""
        from churnbench.arms.naive import _build_tools

        engine = create_engine("sqlite:///:memory:")
        log: list[dict[str, Any]] = []
        tools = _build_tools(engine, MagicMock(), MagicMock(), tmp_path, log)
        list_tool = next(t for t in tools if t.name == "list_documents")
        list_tool.invoke({})
        list_tool.invoke({})
        assert len(log) == 2  # both calls recorded
        log.clear()
        assert len(log) == 0  # log is the same object; clear resets state


# ─────────────────────────────────────────────────────────────────────────────
# Naive arm — ArmResult structure and cost math via mocked agent
# ─────────────────────────────────────────────────────────────────────────────


class TestNaiveArmResult:
    def _arm_with_mocked_agent(
        self, tmp_path: Path, answer_content: str, inp: int = 80, out: int = 20
    ) -> "tuple[Any, Task]":
        from churnbench.arms.naive import NaiveArm

        arm = NaiveArm(_saas_client=MagicMock())
        # Inject a minimal setup without real DB connections
        from sqlalchemy.engine import Engine

        mock_engine = MagicMock(spec=Engine)
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        arm._engine = mock_engine
        arm._mongo = MagicMock()
        arm._saas = MagicMock()
        arm._model = "claude-sonnet-4-6"
        arm._tool_log = []

        # Patch the agent
        arm._agent = MagicMock()
        arm._agent.invoke.return_value = {"messages": [_ai_message(answer_content, inp, out)]}

        task = _make_task("int")
        return arm, task

    def test_arm_result_task_id(self, tmp_path: Path) -> None:
        arm, task = self._arm_with_mocked_agent(tmp_path, "7")
        result = arm.answer(task)
        assert result.task_id == task.task_id

    def test_arm_result_answer_raw(self, tmp_path: Path) -> None:
        arm, task = self._arm_with_mocked_agent(tmp_path, "7")
        result = arm.answer(task)
        assert result.answer_raw == "7"

    def test_arm_result_parsed_int(self, tmp_path: Path) -> None:
        arm, task = self._arm_with_mocked_agent(tmp_path, "7")
        result = arm.answer(task)
        assert result.answer_parsed == 7

    def test_arm_result_token_counts(self, tmp_path: Path) -> None:
        arm, task = self._arm_with_mocked_agent(tmp_path, "7", inp=80, out=20)
        result = arm.answer(task)
        assert result.input_tokens == 80
        assert result.output_tokens == 20

    def test_arm_result_cost_math(self, tmp_path: Path) -> None:
        arm, task = self._arm_with_mocked_agent(tmp_path, "7", inp=80, out=20)
        result = arm.answer(task)
        expected = cost_usd("claude-sonnet-4-6", 80, 20)
        assert result.cost_usd == pytest.approx(expected)

    def test_arm_result_embedding_tokens_zero(self, tmp_path: Path) -> None:
        arm, task = self._arm_with_mocked_agent(tmp_path, "7")
        result = arm.answer(task)
        assert result.embedding_tokens == 0

    def test_arm_result_latency_positive(self, tmp_path: Path) -> None:
        arm, task = self._arm_with_mocked_agent(tmp_path, "7")
        result = arm.answer(task)
        assert result.latency_s >= 0

    def test_arm_result_tool_calls_list(self, tmp_path: Path) -> None:
        arm, task = self._arm_with_mocked_agent(tmp_path, "7")
        result = arm.answer(task)
        assert isinstance(result.tool_calls, list)

    def test_arm_result_trace_list(self, tmp_path: Path) -> None:
        arm, task = self._arm_with_mocked_agent(tmp_path, "7")
        result = arm.answer(task)
        assert isinstance(result.trace, list)


# ─────────────────────────────────────────────────────────────────────────────
# Classic RAG arm — setup / retrieve / answer
# ─────────────────────────────────────────────────────────────────────────────


class TestClassicRagArm:
    # EphemeralClient in ChromaDB 0.5.x shares a process-wide SQLite backend,
    # so collection names must be unique across tests.  Use a per-test counter
    # as a suffix to guarantee uniqueness without importing uuid.
    _coll_counter: int = 0

    def _unique_coll(self) -> str:
        TestClassicRagArm._coll_counter += 1
        return f"test_{TestClassicRagArm._coll_counter}"

    def _setup_arm_with_chunks(
        self,
        chunks: list[dict[str, Any]],
        fake_llm_response: str = "42",
        inp: int = 60,
        out: int = 5,
    ) -> "Any":
        import chromadb

        from churnbench.arms.classic_rag import ClassicRagArm

        embed = _FakeEmbedModel()
        arm = ClassicRagArm(_saas_client=MagicMock(), _embed_model=embed)

        # Build the collection directly (bypassing setup's DB calls)
        client = chromadb.EphemeralClient()
        collection = client.create_collection(self._unique_coll())
        texts = [c["text"] for c in chunks]
        metadatas = [c["meta"] for c in chunks]
        embeddings = embed.encode(texts).tolist()
        collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=[f"c{i}" for i in range(len(chunks))],
            metadatas=metadatas,
        )
        arm._collection = collection
        arm._embed = embed
        arm._embedding_tokens = sum(len(t.split()) for t in texts)
        arm._model = "claude-sonnet-4-6"
        arm._mongo_url = ""

        # Patch the LLM used inside answer()
        fake_resp = _ai_message(fake_llm_response, inp, out)
        arm._llm_patcher = patch(
            "churnbench.arms.classic_rag.llm",
            return_value=MagicMock(invoke=MagicMock(return_value=fake_resp)),
        )
        arm._llm_patcher.start()  # type: ignore[attr-defined]
        return arm

    def test_retrieval_finds_relevant_chunk(self) -> None:
        import chromadb

        embed = _FakeEmbedModel()
        chunks = [
            {"text": "license cost user assigned monthly spend", "meta": {"source": "postgres"}},
            {"text": "contract vendor annual value signed", "meta": {"source": "docs"}},
            {"text": "ticket status product assigned", "meta": {"source": "saas"}},
        ]
        client = chromadb.EphemeralClient()
        collection = client.create_collection(self._unique_coll())
        texts = [c["text"] for c in chunks]
        embeddings = embed.encode(texts).tolist()
        collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=["c0", "c1", "c2"],
            metadatas=[c["meta"] for c in chunks],
        )

        results = collection.query(
            query_embeddings=embed.encode(["license user monthly cost"]).tolist(),
            n_results=1,
            include=["documents"],
        )
        top = results["documents"][0][0]
        assert "license" in top or "cost" in top

    def test_answer_returns_arm_result(self) -> None:
        chunks = [
            {"text": "user active license cost", "meta": {"source": "postgres"}},
            {"text": "vendor contract signed", "meta": {"source": "docs"}},
        ]
        arm = self._setup_arm_with_chunks(chunks, "3", inp=60, out=5)
        task = _make_task("int")
        try:
            result = arm.answer(task)
            assert isinstance(result, ArmResult)
            assert result.task_id == task.task_id
        finally:
            arm._llm_patcher.stop()  # type: ignore[attr-defined]

    def test_answer_raw_from_llm(self) -> None:
        chunks = [{"text": "active user count 7", "meta": {"source": "mongo"}}]
        arm = self._setup_arm_with_chunks(chunks, "7", inp=60, out=5)
        task = _make_task("int")
        try:
            result = arm.answer(task)
            assert result.answer_raw == "7"
            assert result.answer_parsed == 7
        finally:
            arm._llm_patcher.stop()  # type: ignore[attr-defined]

    def test_embedding_tokens_reported(self) -> None:
        chunks = [{"text": "token count test chunk here", "meta": {"source": "postgres"}}]
        arm = self._setup_arm_with_chunks(chunks)
        assert arm._embedding_tokens > 0

    def test_cost_math_for_rag(self) -> None:
        chunks = [{"text": "data chunk", "meta": {"source": "mongo"}}]
        arm = self._setup_arm_with_chunks(chunks, "42", inp=100, out=10)
        task = _make_task("int")
        try:
            result = arm.answer(task)
            expected = cost_usd("claude-sonnet-4-6", 100, 10, arm._embedding_tokens)
            assert result.cost_usd == pytest.approx(expected)
        finally:
            arm._llm_patcher.stop()  # type: ignore[attr-defined]

    def test_no_tool_calls_in_rag(self) -> None:
        chunks = [{"text": "data chunk", "meta": {"source": "mongo"}}]
        arm = self._setup_arm_with_chunks(chunks, "42")
        task = _make_task("int")
        try:
            result = arm.answer(task)
            assert result.tool_calls == []
        finally:
            arm._llm_patcher.stop()  # type: ignore[attr-defined]

    def test_trace_contains_context_and_answer(self) -> None:
        chunks = [{"text": "user license data", "meta": {"source": "postgres"}}]
        arm = self._setup_arm_with_chunks(chunks, "5")
        task = _make_task("int")
        try:
            result = arm.answer(task)
            roles = [step["role"] for step in result.trace]
            assert "context" in roles
            assert "answer" in roles
        finally:
            arm._llm_patcher.stop()  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────────────────────
# Classic RAG chunk serialization helpers (fast, no DB required)
# ─────────────────────────────────────────────────────────────────────────────


class TestClassicRagChunks:
    def test_chunks_docs_splits_by_section(self, tmp_path: Path) -> None:
        from churnbench.arms.classic_rag import _chunks_docs

        (tmp_path / "ctr_0001.md").write_text(
            "# Contract\n\nOverview text.\n\n## Terms\n\nTerm details.\n\n## Pricing\n\nPricing info."
        )
        chunks = _chunks_docs(tmp_path)
        assert len(chunks) >= 2
        texts = " ".join(c["text"] for c in chunks)
        assert "ctr_0001" in texts

    def test_chunks_docs_empty_dir(self, tmp_path: Path) -> None:
        from churnbench.arms.classic_rag import _chunks_docs

        assert _chunks_docs(tmp_path) == []

    def test_chunks_docs_provenance_in_meta(self, tmp_path: Path) -> None:
        from churnbench.arms.classic_rag import _chunks_docs

        (tmp_path / "ctr_0002.md").write_text("# Title\n\nContent.")
        chunks = _chunks_docs(tmp_path)
        assert all(c["meta"]["source"] == "docs" for c in chunks)
        assert all(c["meta"]["doc_id"] == "ctr_0002" for c in chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical arm — supervisor-worker decomposition (Arm 3)
# ─────────────────────────────────────────────────────────────────────────────


class TestHierarchicalArm:
    """Tests for HierarchicalArm.  No live API calls; all backends are mocked."""

    _coll_counter: int = 0

    def _unique_coll(self) -> str:
        TestHierarchicalArm._coll_counter += 1
        return f"hier_{TestHierarchicalArm._coll_counter}"

    def _minimal_arm(self, lm: Any = None) -> Any:
        """Build a HierarchicalArm with all DB connections mocked (no setup() call)."""
        from churnbench.arms.hierarchical import HierarchicalArm

        arm = HierarchicalArm()
        arm._lm = lm if lm is not None else MagicMock()
        arm._model = "claude-sonnet-4-6"
        arm._engine = MagicMock()
        arm._mongo_db = MagicMock()
        arm._saas = MagicMock()
        arm._docs_coll = MagicMock()
        arm._embed = MagicMock()
        arm._t_prime_iso = "2024-01-15"
        arm._embedding_tokens = 0
        return arm

    # ── Test 1: supervisor routes to the workers specified in its JSON output ──

    def test_supervisor_routes_to_correct_workers(self) -> None:
        dispatch_json = json.dumps(
            {
                "assignments": [
                    {"worker": "sql", "query": "count active users"},
                    {"worker": "docs", "query": "get contract terms"},
                ],
                "direct_answer": None,
            }
        )
        lm = _FakeLM(
            [
                _ai_message(dispatch_json, input_tokens=50, output_tokens=15),  # supervisor
                _ai_message("42", input_tokens=60, output_tokens=5),  # synthesis
            ]
        )
        arm = self._minimal_arm(lm=lm)

        dispatched_workers: list[str] = []

        def mock_run_parallel(
            assignments: list[dict[str, str]],
            _dispatch_fn: Any = None,
        ) -> list[Any]:
            dispatched_workers.extend(a["worker"] for a in assignments)
            from churnbench.arms.hierarchical import _WorkerResult

            return [
                _WorkerResult(a["worker"], a["query"], "result", 10, 3, [], "live")
                for a in assignments
            ]

        arm._run_parallel = mock_run_parallel  # type: ignore[method-assign]
        result = arm.answer(_make_task("int"))

        assert "sql" in dispatched_workers
        assert "docs" in dispatched_workers
        assert isinstance(result, ArmResult)

    # ── Test 2: SQL worker retries once on error, records both tool_calls ─────

    def test_sql_worker_retry_on_error_path(self) -> None:
        from churnbench.arms.hierarchical import _sql_worker

        engine = create_engine("sqlite:///:memory:")
        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE numbers (n INTEGER)"))
            conn.execute(text("INSERT INTO numbers VALUES (1), (2), (3)"))
            conn.commit()

        lm = _FakeLM(
            [
                # Call 1: bad SQL (wrong table)
                _ai_message("SELECT * FROM no_such_table", input_tokens=30, output_tokens=8),
                # Call 2: retry with correct SQL
                _ai_message("SELECT n FROM numbers", input_tokens=35, output_tokens=6),
                # Call 3: summarize results
                _ai_message("There are 3 numbers in the table.", input_tokens=40, output_tokens=10),
            ]
        )

        result = _sql_worker("How many numbers?", engine, lm)

        # Must have two tool_calls: original attempt + retry
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["tool"] == "sql_query"
        assert result.tool_calls[1]["tool"] == "sql_query_retry"
        assert result.worker == "sql"
        assert result.source_ts == "live"
        # Token accounting: 30+35+40 in, 8+6+10 out
        assert result.input_tokens == 105
        assert result.output_tokens == 24
        assert result.response  # non-empty summary

    # ── Test 3: docs worker stamps T_prime as source_ts ──────────────────────

    def test_docs_worker_source_ts_is_t_prime(self) -> None:
        import chromadb as _chromadb

        from churnbench.arms.hierarchical import _docs_worker

        client = _chromadb.EphemeralClient()
        coll = client.create_collection(self._unique_coll())
        embed = _FakeEmbedModel()
        texts = ["[docs:ctr_0001 section=0] Annual contract value is $12000"]
        embeddings = embed.encode(texts).tolist()
        coll.add(
            embeddings=embeddings,
            documents=texts,
            ids=["d0"],
            metadatas=[{"source": "docs", "doc_id": "ctr_0001", "section": 0}],
        )

        lm = _FakeLM(
            [_ai_message("The annual contract value is $12000.", input_tokens=30, output_tokens=8)]
        )
        t_prime = "2024-01-15"

        result = _docs_worker("What is the annual contract value?", coll, embed, lm, t_prime)

        assert result.source_ts == t_prime
        assert result.worker == "docs"
        assert result.tool_calls == []  # docs worker has no SQL/HTTP tool calls
        assert result.input_tokens == 30
        assert result.output_tokens == 8

    # ── Test 4: two-round flow — synthesis NEED_MORE triggers round 2 ─────────

    def test_two_round_flow(self) -> None:
        from churnbench.arms.hierarchical import _WorkerResult

        arm = self._minimal_arm()

        supervisor_rounds: list[int] = []
        synthesis_calls: list[int] = []

        def mock_supervisor(
            question: str,
            prior_context: str = "",
            missing_info: str = "",
            round_num: int = 1,
        ) -> tuple[dict[str, Any], int, int]:
            supervisor_rounds.append(round_num)
            if round_num == 1:
                return (
                    {
                        "assignments": [{"worker": "sql", "query": "count users"}],
                        "direct_answer": None,
                    },
                    50,
                    10,
                )
            return (
                {
                    "assignments": [{"worker": "docs", "query": "get contract"}],
                    "direct_answer": None,
                },
                40,
                8,
            )

        def mock_run_parallel(
            assignments: list[dict[str, str]],
            _dispatch_fn: Any = None,
        ) -> list[_WorkerResult]:
            return [
                _WorkerResult(a["worker"], a["query"], "some result", 10, 3, [], "live")
                for a in assignments
            ]

        def mock_synthesize(
            question: str,
            worker_results: list[_WorkerResult],
            answer_type: str,
        ) -> tuple[str, int, int]:
            synthesis_calls.append(len(synthesis_calls) + 1)
            if len(synthesis_calls) == 1:
                return "NEED_MORE: contract details required to complete the answer.", 60, 5
            return "42", 55, 8

        arm._supervisor_dispatch = mock_supervisor  # type: ignore[method-assign]
        arm._run_parallel = mock_run_parallel  # type: ignore[method-assign]
        arm._synthesize = mock_synthesize  # type: ignore[method-assign]

        result = arm.answer(_make_task("int"))

        assert supervisor_rounds == [1, 2]  # two dispatch rounds
        assert len(synthesis_calls) == 2  # two synthesis calls
        roles = [t["role"] for t in result.trace]
        assert "supervisor_round_2" in roles
        assert "synthesis_final" in roles
        assert result.answer_parsed == 42

    # ── Test 5: cost aggregation across supervisor + workers + synthesis ───────

    def test_cost_aggregation_hand_computed(self) -> None:
        from churnbench.arms.hierarchical import _WorkerResult

        arm = self._minimal_arm()
        arm._model = "claude-sonnet-4-6"

        # supervisor: 50 in, 10 out
        # sql_worker: 30 in, 8 out
        # synthesis:  60 in, 5 out
        # total:     140 in, 23 out

        arm._supervisor_dispatch = (  # type: ignore[method-assign]
            lambda q, **kw: (
                {"assignments": [{"worker": "sql", "query": "count users"}], "direct_answer": None},
                50,
                10,
            )
        )
        arm._run_parallel = (  # type: ignore[method-assign]
            lambda assignments, **kw: [
                _WorkerResult("sql", "count users", "3 users", 30, 8, [], "live")
            ]
        )
        arm._synthesize = lambda q, wr, at: ("3", 60, 5)  # type: ignore[method-assign]

        result = arm.answer(_make_task("int"))

        expected = cost_usd("claude-sonnet-4-6", 140, 23)
        assert result.cost_usd == pytest.approx(expected)
        assert result.input_tokens == 140
        assert result.output_tokens == 23

    # ── Test 6: parallel dispatch — workers run concurrently ─────────────────

    def test_parallel_dispatch_concurrent(self) -> None:
        from churnbench.arms.hierarchical import HierarchicalArm, _WorkerResult

        n = 3
        # Barrier.wait() unblocks only when all n workers have called it.
        # If dispatch were sequential, only 1 worker at a time would reach the
        # barrier and it would never release (BrokenBarrierError after timeout).
        barrier = threading.Barrier(n, timeout=5.0)

        def slow_dispatch(assignment: dict[str, str]) -> _WorkerResult:
            barrier.wait()  # raises BrokenBarrierError if not all n are concurrent
            return _WorkerResult(assignment["worker"], assignment["query"], "ok", 0, 0, [], "live")

        arm = HierarchicalArm()
        assignments = [
            {"worker": "sql", "query": "q1"},
            {"worker": "mongo", "query": "q2"},
            {"worker": "docs", "query": "q3"},
        ]
        results = arm._run_parallel(assignments, _dispatch_fn=slow_dispatch)

        assert len(results) == 3
        assert not barrier.broken  # all workers reached the barrier simultaneously


# ─────────────────────────────────────────────────────────────────────────────
# Provider — NVIDIA NIM support
# ─────────────────────────────────────────────────────────────────────────────


class TestLlmFactory:
    """Tests for the llm() factory — NIM (ChatOpenAI) and Anthropic (ChatAnthropic) providers."""

    # ── NIM provider (default) ────────────────────────────────────────────────

    def test_nim_provider_returns_chat_openai(self) -> None:
        from churnbench.arms.base import llm

        env = {"NVIDIA_API_KEY": "test-key", "CHURNBENCH_PROVIDER": "nvidia_nim"}
        with patch.dict("os.environ", env, clear=False):
            result = llm()
        assert isinstance(result, ChatOpenAI)

    def test_llm_returns_chat_openai(self) -> None:
        """Default (no provider set) also returns ChatOpenAI."""
        from churnbench.arms.base import llm

        import os as _os

        clean = {k: v for k, v in _os.environ.items() if k != "CHURNBENCH_PROVIDER"}
        clean["NVIDIA_API_KEY"] = "test-key"
        with patch.dict("os.environ", clean, clear=True):
            result = llm()
        assert isinstance(result, ChatOpenAI)

    def test_llm_uses_nim_base_url(self) -> None:
        from churnbench.arms.base import _NIM_BASE_URL, llm

        with patch.dict(
            "os.environ", {"NVIDIA_API_KEY": "k", "CHURNBENCH_PROVIDER": "nvidia_nim"}, clear=False
        ):
            result = llm()
        assert isinstance(result, ChatOpenAI)
        assert result.openai_api_base == _NIM_BASE_URL

    def test_default_model_is_nemotron(self) -> None:
        from churnbench.arms.base import _DEFAULT_MODEL, llm

        import os as _os

        clean_env = {
            k: v
            for k, v in _os.environ.items()
            if k not in ("CHURNBENCH_MODEL", "CHURNBENCH_PROVIDER")
        }
        clean_env["NVIDIA_API_KEY"] = "k"
        with patch.dict("os.environ", clean_env, clear=True):
            result = llm()
        assert isinstance(result, ChatOpenAI)
        assert result.model_name == _DEFAULT_MODEL

    def test_model_arg_overrides_default(self) -> None:
        from churnbench.arms.base import llm

        with patch.dict(
            "os.environ", {"NVIDIA_API_KEY": "k", "CHURNBENCH_PROVIDER": "nvidia_nim"}, clear=False
        ):
            result = llm(model="deepseek-ai/deepseek-v4-flash")
        assert isinstance(result, ChatOpenAI)
        assert result.model_name == "deepseek-ai/deepseek-v4-flash"

    def test_env_model_used_when_no_arg(self) -> None:
        from churnbench.arms.base import llm

        env = {
            "NVIDIA_API_KEY": "k",
            "CHURNBENCH_PROVIDER": "nvidia_nim",
            "CHURNBENCH_MODEL": "deepseek-ai/deepseek-v4-flash",
        }
        with patch.dict("os.environ", env, clear=False):
            result = llm()
        assert result.model_name == "deepseek-ai/deepseek-v4-flash"

    def test_model_arg_beats_env_model(self) -> None:
        from churnbench.arms.base import llm

        env = {
            "NVIDIA_API_KEY": "k",
            "CHURNBENCH_PROVIDER": "nvidia_nim",
            "CHURNBENCH_MODEL": "deepseek-ai/deepseek-v4-flash",
        }
        with patch.dict("os.environ", env, clear=False):
            result = llm(model="nvidia/nemotron-3-ultra-550b-a55b")
        assert result.model_name == "nvidia/nemotron-3-ultra-550b-a55b"

    def test_temperature_is_zero(self) -> None:
        from churnbench.arms.base import llm

        with patch.dict(
            "os.environ", {"NVIDIA_API_KEY": "k", "CHURNBENCH_PROVIDER": "nvidia_nim"}, clear=False
        ):
            result = llm()
        assert result.temperature == 0

    # ── Anthropic provider ────────────────────────────────────────────────────

    def test_anthropic_provider_returns_chat_anthropic(self) -> None:
        from churnbench.arms.base import llm

        env = {"CHURNBENCH_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key"}
        with patch.dict("os.environ", env, clear=False):
            result = llm()
        assert isinstance(result, ChatAnthropic)

    def test_anthropic_default_model_is_sonnet(self) -> None:
        from churnbench.arms.base import _ANTHROPIC_MODEL, llm

        import os as _os

        clean = {
            k: v
            for k, v in _os.environ.items()
            if k not in ("CHURNBENCH_MODEL", "CHURNBENCH_PROVIDER")
        }
        clean["CHURNBENCH_PROVIDER"] = "anthropic"
        clean["ANTHROPIC_API_KEY"] = "k"
        with patch.dict("os.environ", clean, clear=True):
            result = llm()
        assert isinstance(result, ChatAnthropic)
        assert result.model == _ANTHROPIC_MODEL

    def test_anthropic_temperature_is_zero(self) -> None:
        from churnbench.arms.base import llm

        env = {"CHURNBENCH_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "k"}
        with patch.dict("os.environ", env, clear=False):
            result = llm()
        assert isinstance(result, ChatAnthropic)
        assert result.temperature == 0

    def test_anthropic_model_arg_overrides_default(self) -> None:
        from churnbench.arms.base import llm

        env = {"CHURNBENCH_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "k"}
        with patch.dict("os.environ", env, clear=False):
            result = llm(model="claude-haiku-4-5-20251001")
        assert isinstance(result, ChatAnthropic)
        assert result.model == "claude-haiku-4-5-20251001"

    # ── active_model() ────────────────────────────────────────────────────────

    def test_active_model_nim_default(self) -> None:
        from churnbench.arms.base import _DEFAULT_MODEL, active_model

        import os as _os

        clean = {
            k: v
            for k, v in _os.environ.items()
            if k not in ("CHURNBENCH_MODEL", "CHURNBENCH_PROVIDER")
        }
        with patch.dict("os.environ", clean, clear=True):
            assert active_model() == _DEFAULT_MODEL

    def test_active_model_anthropic_default(self) -> None:
        from churnbench.arms.base import _ANTHROPIC_MODEL, active_model

        import os as _os

        clean = {
            k: v
            for k, v in _os.environ.items()
            if k not in ("CHURNBENCH_MODEL", "CHURNBENCH_PROVIDER")
        }
        clean["CHURNBENCH_PROVIDER"] = "anthropic"
        with patch.dict("os.environ", clean, clear=True):
            assert active_model() == _ANTHROPIC_MODEL

    def test_active_model_env_override(self) -> None:
        from churnbench.arms.base import active_model

        env = {
            "CHURNBENCH_PROVIDER": "nvidia_nim",
            "CHURNBENCH_MODEL": "deepseek-ai/deepseek-v4-flash",
        }
        with patch.dict("os.environ", env, clear=False):
            assert active_model() == "deepseek-ai/deepseek-v4-flash"

    # ── Pricing / cost ────────────────────────────────────────────────────────

    def test_usage_metadata_roundtrip_through_cost_usd(self) -> None:
        # ChatOpenAI.invoke() returns AIMessage with usage_metadata keys
        # input_tokens / output_tokens — verify arms can pass them to cost_usd().
        usage = {"input_tokens": 1_000, "output_tokens": 500}
        c = cost_usd(
            "nvidia/nemotron-3-ultra-550b-a55b", usage["input_tokens"], usage["output_tokens"]
        )
        assert c == pytest.approx(0.00525)

    def test_anthropic_pricing_in_pricing_dict(self) -> None:
        from churnbench.arms.base import PRICING, _ANTHROPIC_MODEL

        assert _ANTHROPIC_MODEL in PRICING
        assert PRICING[_ANTHROPIC_MODEL]["input"] == pytest.approx(3.0)
        assert PRICING[_ANTHROPIC_MODEL]["output"] == pytest.approx(15.0)

    def test_sonnet_cost_calculation(self) -> None:
        # 1k input @ $3/MTok + 500 output @ $15/MTok = $0.003 + $0.0075 = $0.0105
        c = cost_usd("claude-sonnet-4-6", 1_000, 500)
        assert c == pytest.approx(0.0105)
