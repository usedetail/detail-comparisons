"""Per-repo tournament: filter, summarize, judge, rank."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from review_bot_comparison.bradley_terry import fit_bradley_terry
from review_bot_comparison.llm import is_real_finding, judge_pair, summarize_finding

if TYPE_CHECKING:
    import random

    import anthropic

    from review_bot_comparison.cache import DiskCache

logger = logging.getLogger(__name__)

DIFF_HUNK_CHAR_LIMIT = 500


def format_bot_comment(comment: dict[str, Any]) -> str:
    parts: list[str] = []
    if path := comment.get("comment_path"):
        parts.append(f"File: {path}")
    if hunk := comment.get("comment_diff_hunk"):
        parts.append(hunk[:DIFF_HUNK_CHAR_LIMIT])
    if body := comment.get("comment_body"):
        parts.append(body)
    return "\n".join(parts)


def format_detail_finding(finding: dict[str, Any]) -> str:
    parts: list[str] = []
    if path := finding.get("file_path"):
        parts.append(f"File: {path}")
    if title := finding.get("title"):
        parts.append(title)
    if summary := finding.get("summary"):
        parts.append(summary)
    return "\n".join(parts)


def _filter_bot_comments(
    bot_comments: list[dict[str, Any]],
    client: anthropic.Anthropic,
    filter_cache: DiskCache,
    workers: int,
) -> list[tuple[dict[str, Any], str]]:
    """Drop greetings/boilerplate; return (comment, formatted_text) for real findings."""
    texts = [format_bot_comment(c) for c in bot_comments]
    keep = [False] * len(texts)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(is_real_finding, client, text, filter_cache): idx
            for idx, text in enumerate(texts)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Filtering"):
            keep[futures[future]] = future.result()
    filter_cache.save()
    return [(bot_comments[i], texts[i]) for i in range(len(bot_comments)) if keep[i]]


def _summarize_all(
    items: list[dict[str, Any]],
    client: anthropic.Anthropic,
    summary_cache: DiskCache,
    workers: int,
) -> None:
    """Add a "summary" key to each item, in place."""
    summaries: list[str | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(summarize_finding, client, item["raw"], summary_cache): idx
            for idx, item in enumerate(items)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Summarizing"):
            summaries[futures[future]] = future.result()
    summary_cache.save()
    for idx, summary in enumerate(summaries):
        items[idx]["summary"] = summary or ""


def _judge_matchups(
    items: list[dict[str, Any]],
    matchups: list[tuple[int, int]],
    swaps: list[bool],
    client: anthropic.Anthropic,
    judgment_cache: DiskCache,
    workers: int,
) -> list[tuple[int, int]]:
    """Judge all matchups in parallel; return (winner_idx, loser_idx) pairs."""
    outcomes: list[tuple[int, int]] = []

    def _judge(idx: int) -> tuple[int, int]:
        i, j = matchups[idx]
        winner, _ = judge_pair(
            client, items[i]["summary"], items[j]["summary"], swaps[idx], judgment_cache
        )
        return (i, j) if winner == "a" else (j, i)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_judge, idx) for idx in range(len(matchups))]
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Judging"):
            outcomes.append(future.result())
    judgment_cache.save()
    return outcomes


def run_repo_tournament(
    repo: str,
    bot_name: str,
    bot_comments: list[dict[str, Any]],
    detail_findings: list[dict[str, Any]],
    client: anthropic.Anthropic,
    rng: random.Random,
    filter_cache: DiskCache,
    summary_cache: DiskCache,
    judgment_cache: DiskCache,
    n_comparisons: int = 0,
    workers: int = 8,
) -> dict[str, Any]:
    """Run a tournament for one (repo, bot) pair: filter, summarize, judge, rank.

    Returns a result dict with overall counts and a per-finding ranking.
    """
    logger.info("\n%s", "=" * 60)
    logger.info(
        "  %s: detail (%d) vs %s (%d)",
        repo,
        len(detail_findings),
        bot_name,
        len(bot_comments),
    )
    logger.info("%s", "=" * 60)

    logger.info("  Filtering %d bot comments...", len(bot_comments))
    filtered_bot = _filter_bot_comments(bot_comments, client, filter_cache, workers)
    logger.info("  Kept %d/%d real findings", len(filtered_bot), len(bot_comments))

    if not filtered_bot:
        logger.info("  No real bot findings — skipping")
        return {"repo": repo, "bot": bot_name, "detail_wins": 0, "bot_wins": 0, "total": 0}

    items: list[dict[str, Any]] = [
        {"tool": "detail", "raw": format_detail_finding(f)} for f in detail_findings
    ]
    items.extend({"tool": bot_name, "raw": raw} for _, raw in filtered_bot)

    logger.info("  Summarizing %d findings...", len(items))
    _summarize_all(items, client, summary_cache, workers)

    n = len(items)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    target = n_comparisons if n_comparisons > 0 else 5 * n
    matchups = rng.sample(all_pairs, min(target, len(all_pairs)))
    swaps = [rng.random() < 0.5 for _ in matchups]  # pre-decided so judging is deterministic

    logger.info("  Judging %d matchups...", len(matchups))
    outcomes = _judge_matchups(items, matchups, swaps, client, judgment_cache, workers)

    strengths = fit_bradley_terry(n, outcomes)
    ranked = sorted(range(n), key=lambda x: -strengths[x])
    rank_of = {idx: rank for rank, idx in enumerate(ranked)}

    detail_wins = sum(1 for w, _ in outcomes if items[w]["tool"] == "detail")
    total = len(outcomes)
    logger.info("\n  Detail wins: %d/%d (%.0f%%)", detail_wins, total, 100 * detail_wins / total)

    finding_ranks = sorted(
        (
            {
                "tool": items[idx]["tool"],
                "summary": items[idx]["summary"],
                "bt_score": round(strengths[idx], 4),
                "rank": rank_of[idx] + 1,
                "percentile": round(100 * (1 - rank_of[idx] / max(n - 1, 1)), 1),
            }
            for idx in range(n)
        ),
        key=lambda x: x["rank"],
    )

    logger.info("  Top 10 findings:")
    for fr in finding_ranks[:10]:
        tool_label = "Detail" if fr["tool"] == "detail" else bot_name
        summary_excerpt = str(fr["summary"])[:60]
        logger.info(
            "    #%d [%s] (BT=%s) %s",
            fr["rank"],
            tool_label,
            fr["bt_score"],
            summary_excerpt,
        )

    return {
        "repo": repo,
        "bot": bot_name,
        "detail_wins": detail_wins,
        "bot_wins": total - detail_wins,
        "total": total,
        "finding_ranks": finding_ranks,
    }
