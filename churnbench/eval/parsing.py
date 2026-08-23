"""Parse raw LLM string output to the task's answer_type.

Strict but fair:
  - "3 licenses" → 3              (int)
  - "$1,234.56" → 1234.56         (float)
  - '["A", "B"]' → ["A", "B"]    (list[str])
  - "active" → "active"           (str)

Parse failures are returned with parse_failure=True.  Arms and the scorer
track these separately so we can report them in the paper (parse failures ≠
wrong answers — they indicate a format mismatch, not content mismatch).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ParseResult:
    """Outcome of parsing a raw LLM string to a typed answer."""

    value: Any  # None when parse_failure is True
    parse_failure: bool
    raw: str


def parse(raw: str, answer_type: str) -> ParseResult:
    """Parse raw LLM output to the task's answer_type.

    Always returns a ParseResult; callers check parse_failure.
    """
    raw = raw.strip()
    match answer_type:
        case "int":
            return _parse_int(raw)
        case "float":
            return _parse_float(raw)
        case "str":
            return _parse_str(raw)
        case "list[str]":
            return _parse_list_str(raw)
        case _:
            return _parse_str(raw)


# ── Per-type parsers ─────────────────────────────────────────────────────────


def _parse_int(raw: str) -> ParseResult:
    """Extract the first integer from raw; fail on empty / sentinel strings."""
    lowered = raw.lower().strip()
    if lowered in {"none", "null", "n/a", "na", "not found", "not_found", ""}:
        return ParseResult(value=None, parse_failure=True, raw=raw)
    # Remove thousands separators and currency symbols before matching
    cleaned = raw.replace(",", "").replace("$", "")
    m = re.search(r"-?\d+", cleaned)
    if m:
        return ParseResult(value=int(m.group()), parse_failure=False, raw=raw)
    return ParseResult(value=None, parse_failure=True, raw=raw)


def _parse_float(raw: str) -> ParseResult:
    """Extract the first decimal number from raw; handle $, commas, %."""
    lowered = raw.lower().strip()
    if lowered in {"none", "null", "n/a", "na", "not found", "not_found", ""}:
        return ParseResult(value=None, parse_failure=True, raw=raw)
    # Remove thousands separators, currency symbols, percent signs
    cleaned = re.sub(r"[,$%]", "", raw)
    m = re.search(r"-?\d+\.?\d*", cleaned)
    if m:
        return ParseResult(value=float(m.group()), parse_failure=False, raw=raw)
    return ParseResult(value=None, parse_failure=True, raw=raw)


def _parse_str(raw: str) -> ParseResult:
    """Strip outer whitespace, markdown formatting, and surrounding quotes."""
    # Strip markdown bold/italic/code markers (* and `).
    # Do NOT include _ — entity IDs like cc_008 and prd_0005 contain underscores.
    val = re.sub(r"[*`]+", "", raw).strip()
    # Strip surrounding single/double quotes
    if len(val) >= 2 and val[0] in "\"'`" and val[-1] == val[0]:
        val = val[1:-1]
    return ParseResult(value=val, parse_failure=False, raw=raw)


def _parse_list_str(raw: str) -> ParseResult:
    """Parse a JSON array or comma/newline-separated list of strings."""
    stripped = raw.strip()
    if not stripped:
        return ParseResult(value=None, parse_failure=True, raw=raw)

    # Try JSON first
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            items = [str(v).strip() for v in parsed]
            return ParseResult(value=items, parse_failure=False, raw=raw)
    except json.JSONDecodeError:
        pass

    # Try bracket-wrapped but with single quotes (LLMs sometimes do this)
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1]
        items = [s.strip().strip("'\"`") for s in re.split(r",\s*", inner) if s.strip()]
        if items:
            return ParseResult(value=items, parse_failure=False, raw=raw)

    # Fall back to newline or comma separation
    if "\n" in stripped:
        items = [s.strip().strip("-•*\t '\"`") for s in stripped.split("\n") if s.strip()]
    else:
        items = [s.strip().strip("'\"`") for s in stripped.split(",") if s.strip()]

    if items:
        return ParseResult(value=items, parse_failure=False, raw=raw)

    return ParseResult(value=None, parse_failure=True, raw=raw)
