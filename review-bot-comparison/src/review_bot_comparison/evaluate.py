"""Run the Detail-vs-bot tournament across all configured (repo, bot) pairs.

Reads bot comments from `data/wild_reviews.json` and Detail findings from
`data/detail_findings.json`. For each repo, runs `run_repo_tournament` and
writes the combined ranking to `data/eval_ranking.json`.

Usage:
    uv run evaluate [--seed 42] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import anthropic

from review_bot_comparison.cache import DATA_DIR, DiskCache
from review_bot_comparison.tournament import run_repo_tournament

logger = logging.getLogger(__name__)

WILD_REVIEWS_PATH = DATA_DIR / "wild_reviews.json"
DETAIL_FINDINGS_PATH = DATA_DIR / "detail_findings.json"
DEFAULT_OUTPUT_PATH = DATA_DIR / "eval_ranking.json"


def _load_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with WILD_REVIEWS_PATH.open() as f:
        wild_reviews = json.load(f)
    with DETAIL_FINDINGS_PATH.open() as f:
        detail_findings = json.load(f)
    return wild_reviews, detail_findings


def _group_by_repo(
    wild_reviews: list[dict[str, Any]],
    detail_findings: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, list[dict[str, Any]]]]:
    bot_by_repo: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rec in wild_reviews:
        bot_by_repo[rec["repo"]][rec["tool"]].append(rec)

    detail_by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in detail_findings:
        detail_by_repo[rec["repo"]].append(rec)

    return bot_by_repo, detail_by_repo


def _log_summary(results: list[dict[str, Any]]) -> None:
    total_dw = sum(r["detail_wins"] for r in results)
    total_bw = sum(r["bot_wins"] for r in results)

    logger.info("\n%s", "=" * 60)
    logger.info("  RESULTS (%d matchups)", len(results))
    logger.info("%s", "=" * 60)
    for r in results:
        pct = 100 * r["detail_wins"] / r["total"] if r["total"] else 0
        logger.info("  %-40s vs %-35s %.0f%% detail", r["repo"], r["bot"], pct)

    if total_dw + total_bw > 0:
        share = 100 * total_dw / (total_dw + total_bw)
        logger.info("\n  Overall: %d/%d (%.0f%%)", total_dw, total_dw + total_bw, share)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Detail vs AI review bots")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    # Clean format so tournament's logger.info() reads like CLI output;
    # --verbose adds level/timestamps for debugging.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=(
            "%(asctime)s %(levelname)s %(name)s: %(message)s" if args.verbose else "%(message)s"
        ),
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable is required.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    rng = random.Random(args.seed)

    filter_cache = DiskCache("filters")
    summary_cache = DiskCache("summaries")
    judgment_cache = DiskCache("judgments")
    for cache in (filter_cache, summary_cache, judgment_cache):
        cache.load()

    wild_reviews, detail_findings = _load_inputs()
    bot_by_repo, detail_by_repo = _group_by_repo(wild_reviews, detail_findings)

    results: list[dict[str, Any]] = []
    for repo, findings in sorted(detail_by_repo.items()):
        for bot_name, bot_comments in sorted(bot_by_repo.get(repo, {}).items()):
            if not bot_comments:
                continue
            results.append(
                run_repo_tournament(
                    repo=repo,
                    bot_name=bot_name,
                    bot_comments=bot_comments,
                    detail_findings=findings,
                    client=client,
                    rng=rng,
                    filter_cache=filter_cache,
                    summary_cache=summary_cache,
                    judgment_cache=judgment_cache,
                    workers=args.workers,
                )
            )

    _log_summary(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("\n  Results saved to %s", args.output)


if __name__ == "__main__":
    main()
