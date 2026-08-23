"""Base classes and shared infrastructure for all ChurnBench arms.

Paper note (§4 limitations / future work):
  The simulator's offboard-unassigns-licenses invariant means "orphaned license"
  scenarios cannot exist in simulated data.  Real enterprises do have orphaned
  licenses (a trap type in prior SaaS benchmarks), so the simulator is slightly
  idealized here.  A future drift stream that adds "offboard without cleanup" as
  a failure mode would address this gap.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from churnbench.tasks.schema import Task


# ── Provider / endpoint constants ─────────────────────────────────────────────
_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
_DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
_ANTHROPIC_MODEL = "claude-sonnet-4-6"


# ── Pricing ───────────────────────────────────────────────────────────────────
# $/million tokens
PRICING: dict[str, dict[str, float]] = {
    # Source: https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b (Pricing tab)
    "nvidia/nemotron-3-ultra-550b-a55b": {"input": 3.50, "output": 3.50},
    # Source: https://build.nvidia.com/deepseek-ai/deepseek-v4-flash (Pricing tab)
    # TODO: verify — page is JS-rendered and could not be scraped automatically
    "deepseek-ai/deepseek-v4-flash": {"input": 0.20, "output": 0.60},
    # Source: https://www.anthropic.com/pricing (Claude Sonnet 4.6)
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def active_model() -> str:
    """Return the active model name for the current provider.

    Priority:
        1. ``CHURNBENCH_MODEL`` env var (explicit override, any provider)
        2. Provider default: ``_ANTHROPIC_MODEL`` for anthropic, else ``_DEFAULT_MODEL``
    """
    provider = os.environ.get("CHURNBENCH_PROVIDER", "nvidia_nim")
    default = _ANTHROPIC_MODEL if provider == "anthropic" else _DEFAULT_MODEL
    return os.environ.get("CHURNBENCH_MODEL", default)


def llm(model: str | None = None) -> BaseChatModel:
    """Return a zero-temperature chat model for the configured provider.

    Provider selection: ``CHURNBENCH_PROVIDER`` env var
        ``nvidia_nim`` (default) → ``ChatOpenAI`` via NVIDIA NIM API
        ``anthropic``            → ``ChatAnthropic``

    Model selection priority:
        1. ``model`` argument
        2. ``CHURNBENCH_MODEL`` env var
        3. Provider default (``_DEFAULT_MODEL`` for NIM, ``_ANTHROPIC_MODEL`` for Anthropic)

    Requires ``NVIDIA_API_KEY`` for nvidia_nim, or ``ANTHROPIC_API_KEY`` for anthropic.
    All arms call this helper so provider and sampling are never confounds across arms.
    """
    provider = os.environ.get("CHURNBENCH_PROVIDER", "nvidia_nim")

    if provider == "anthropic":
        _model = model or os.environ.get("CHURNBENCH_MODEL", _ANTHROPIC_MODEL)
        return ChatAnthropic(
            model=_model,
            api_key=SecretStr(os.environ.get("ANTHROPIC_API_KEY", "")),
            temperature=0,
            max_retries=2,
        )

    # Default: nvidia_nim
    _model = model or os.environ.get("CHURNBENCH_MODEL", _DEFAULT_MODEL)
    return ChatOpenAI(
        model=_model,
        base_url=_NIM_BASE_URL,
        api_key=SecretStr(os.environ.get("NVIDIA_API_KEY", "")),
        temperature=0,
        max_retries=2,
    )


# ── Cost accounting ───────────────────────────────────────────────────────────


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    embedding_tokens: int = 0,
) -> float:
    """Single place for cost arithmetic — no scattered math in arm code.

    The model string selects the right entry in PRICING.  Unknown models fall
    back to the nemotron rate (conservative over-estimate for unrecognised NIM
    endpoints) so costs never silently become zero.

    Embedding cost is always $0 for local sentence-transformers; embedding_tokens
    is still accepted here so call-sites are uniform and the cost-decomposition
    table can report token counts for each component.
    """
    pricing = PRICING.get(model, PRICING[_DEFAULT_MODEL])
    lm_cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
    # embedding_cost = 0.0 — local model, no API fee
    return round(lm_cost, 8)


# ── Shared dataclasses ────────────────────────────────────────────────────────


@dataclass
class FabricConfig:
    """Connection coordinates for the four fabric layers."""

    pg_url: str = "postgresql://churn:churn@localhost:5433/sam_warehouse"
    mongo_url: str = "mongodb://localhost:27018"
    saas_base_url: str = "http://localhost:8010"
    docs_dir: Path = field(default_factory=lambda: Path("docs"))


@dataclass
class ArmResult:
    """One arm's response to one task — all data needed to score and attribute."""

    task_id: str
    answer_raw: str
    answer_parsed: Any  # coerced to the task's answer_type by eval/parsing.py
    input_tokens: int
    output_tokens: int
    embedding_tokens: int  # non-zero only for classic_rag; always $0 cost
    cost_usd: float
    latency_s: float
    tool_calls: list[dict[str, Any]]  # [{tool, target, duration_ms}]
    trace: list[dict[str, Any]]  # messages/steps; used for freshness-error attribution


# ── Abstract base ─────────────────────────────────────────────────────────────


class BaseArm(ABC):
    """Abstract arm interface.

    Lifecycle::

        arm = ConcreteArm()
        arm.setup(config, T_prime)          # build caches at T_prime — staleness born here
        results = [arm.answer(t) for t in tasks]  # tasks evaluated at T
        arm.teardown()

    setup() and answer() are always called on the same instance in sequence.
    teardown() must be safe to call even if setup() was never called.
    """

    @abstractmethod
    def setup(self, config: FabricConfig, T_prime: date) -> None:
        """Build indexes, caches, or DB connections at T_prime."""

    @abstractmethod
    def answer(self, task: Task) -> ArmResult:
        """Answer a single task; return a fully-populated ArmResult."""

    def teardown(self) -> None:
        """Release connections and temp files.  Default: no-op."""
