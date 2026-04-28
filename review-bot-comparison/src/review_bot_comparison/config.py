"""Evaluation configuration: which bots, repos, and time window we evaluate over."""

from __future__ import annotations

# Inclusive start, exclusive end. ISO-8601 dates. The week of bot review activity
# we evaluate against — Detail is also configured to scan the same window.
EVAL_WEEK: tuple[str, str] = ("2026-03-17", "2026-03-24")

# Each bot is paired with the repos it actively reviews on.
EVAL_BOTS: dict[str, list[str]] = {
    "chatgpt-codex-connector[bot]": ["openclaw/openclaw"],
    "gemini-code-assist[bot]": ["vllm-project/vllm"],
}

EVAL_REPOS: list[str] = sorted({repo for repos in EVAL_BOTS.values() for repo in repos})

# Anthropic model used for filtering, summarization, and pairwise judging.
MODEL: str = "claude-sonnet-4-6"
