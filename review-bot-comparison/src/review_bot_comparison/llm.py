"""Anthropic API wrappers: filter, summarize, pairwise judge.

All three LLM operations follow the same pattern — tool-call shape, cached by
hash of (prompt + input). Caches are passed in by the caller so this module
has no module-level state.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import anthropic

from review_bot_comparison.cache import text_hash
from review_bot_comparison.config import MODEL

if TYPE_CHECKING:
    from collections.abc import Callable

    from review_bot_comparison.cache import DiskCache

logger = logging.getLogger(__name__)

MAX_RETRIES = 10
RETRY_BASE_DELAY = 15
INPUT_CHAR_LIMIT = 2000


def _call_with_retry[T](fn: Callable[[], T]) -> T:
    """Retry transient API errors with exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except anthropic.APIStatusError as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2**attempt)
            logger.warning(
                "API error (attempt %d/%d), retrying in %ds: %s",
                attempt + 1,
                MAX_RETRIES,
                delay,
                exc,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable: retry loop exited without returning or raising")


def _extract_tool_input(resp: anthropic.types.Message, tool_name: str) -> dict | None:
    """Return the input dict from the first tool_use block matching tool_name, or None."""
    for block in resp.content:
        if isinstance(block, anthropic.types.ToolUseBlock) and block.name == tool_name:
            return block.input if isinstance(block.input, dict) else None
    return None


# ── Filter: classify whether a comment is a real finding ─────────────────────

FILTER_PROMPT = """\
Is this an actual code review comment about specific code? \
Answer false for greetings and empty messages.

{comment}\
"""

FILTER_TOOL: anthropic.types.ToolParam = {
    "name": "classify",
    "description": "Classify whether this is a real code review finding.",
    "input_schema": {
        "type": "object",
        "properties": {
            "is_finding": {
                "type": "boolean",
                "description": "true if this is a real finding, false if boilerplate.",
            },
        },
        "required": ["is_finding"],
    },
}


def is_real_finding(client: anthropic.Anthropic, text: str, cache: DiskCache) -> bool:
    """Classify a comment as a real finding (vs. greeting/boilerplate). Cached."""
    key = text_hash(FILTER_PROMPT + text)
    if key in cache:
        return bool(cache.get(key))

    resp = _call_with_retry(
        lambda: client.messages.create(
            model=MODEL,
            max_tokens=64,
            tools=[FILTER_TOOL],
            tool_choice={"type": "tool", "name": "classify"},
            messages=[
                {"role": "user", "content": FILTER_PROMPT.format(comment=text[:INPUT_CHAR_LIMIT])}
            ],
        )
    )
    tool_input = _extract_tool_input(resp, "classify") or {}
    result = bool(tool_input.get("is_finding", False))
    cache.set(key, result)
    return result


# ── Summarize: one-line description ──────────────────────────────────────────

SUMMARIZE_PROMPT = """\
Summarize this code review finding in one sentence.

{finding}\
"""

SUMMARIZE_TOOL: anthropic.types.ToolParam = {
    "name": "summarize",
    "description": "Provide a one-sentence summary of the code review finding.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "One sentence describing the bug or issue and what could go wrong.",
            },
        },
        "required": ["summary"],
    },
}


def summarize_finding(client: anthropic.Anthropic, text: str, cache: DiskCache) -> str:
    """One-sentence summary of a finding via tool use. Cached."""
    key = text_hash(SUMMARIZE_PROMPT + text)
    if key in cache:
        return str(cache.get(key))

    resp = _call_with_retry(
        lambda: client.messages.create(
            model=MODEL,
            max_tokens=200,
            tools=[SUMMARIZE_TOOL],
            tool_choice={"type": "tool", "name": "summarize"},
            messages=[
                {
                    "role": "user",
                    "content": SUMMARIZE_PROMPT.format(finding=text[:INPUT_CHAR_LIMIT]),
                }
            ],
        )
    )
    tool_input = _extract_tool_input(resp, "summarize") or {}
    result = str(tool_input.get("summary", ""))
    cache.set(key, result)
    return result


# ── Judge: pick the more important of two summaries ──────────────────────────

JUDGE_PROMPT = """\
You are comparing two bug findings from code review tools on the same codebase.

An engineer only has time to investigate one. Pick the one that is more \
important for them to see.

Finding A: {summary_a}

Finding B: {summary_b}

Use the select_winner tool to pick a or b.\
"""

SELECT_WINNER_TOOL: anthropic.types.ToolParam = {
    "name": "select_winner",
    "description": "Select which finding (a or b) is more important.",
    "input_schema": {
        "type": "object",
        "properties": {
            "justification": {
                "type": "string",
                "description": "One sentence explaining why the winner is more important.",
            },
            "winner": {
                "type": "string",
                "enum": ["a", "b"],
            },
        },
        "required": ["justification", "winner"],
    },
}


def _judgment_cache_key(summary_a: str, summary_b: str) -> str:
    """Order-independent cache key for a pair."""
    pair = sorted([summary_a, summary_b])
    return text_hash(JUDGE_PROMPT + pair[0] + "||" + pair[1])


def judge_pair(
    client: anthropic.Anthropic,
    summary_a: str,
    summary_b: str,
    swap: bool,
    cache: DiskCache,
) -> tuple[str, str]:
    """Compare two summaries. Returns (winner 'a'|'b', justification). Cached.

    `swap` controls presentation order to mitigate position bias — caller decides
    deterministically (e.g. seeded RNG) so judging is reproducible across workers.
    """
    key = _judgment_cache_key(summary_a, summary_b)
    cached = cache.get(key)
    if isinstance(cached, dict):
        winner_summary = str(cached.get("winner_summary", ""))
        justification = str(cached.get("justification", ""))
        return ("a" if winner_summary == summary_a else "b"), justification

    first, second = (summary_b, summary_a) if swap else (summary_a, summary_b)
    resp = _call_with_retry(
        lambda: client.messages.create(
            model=MODEL,
            max_tokens=200,
            tools=[SELECT_WINNER_TOOL],
            tool_choice={"type": "tool", "name": "select_winner"},
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(summary_a=first, summary_b=second),
                }
            ],
        )
    )

    tool_input = _extract_tool_input(resp, "select_winner") or {}
    winner_label = str(tool_input.get("winner", "a"))
    justification = str(tool_input.get("justification", ""))

    if swap:
        winner_label = "b" if winner_label == "a" else "a"

    winner_summary = summary_a if winner_label == "a" else summary_b
    cache.set(key, {"winner_summary": winner_summary, "justification": justification})

    return winner_label, justification
