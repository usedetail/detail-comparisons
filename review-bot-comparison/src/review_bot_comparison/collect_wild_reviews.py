"""Collect AI bot review comments from GHArchive via the ClickHouse playground.

The playground exposes the GHArchive event stream as a SQL-queryable table
(`github_events`). We pull review-style events authored by the configured bots
on the configured repos within `EVAL_WEEK`, drop pairs with too few comments
to be meaningful, and write the result to `data/wild_reviews.json`.

Usage:
    uv run collect-wild-reviews
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from review_bot_comparison.cache import DATA_DIR
from review_bot_comparison.config import EVAL_BOTS, EVAL_REPOS, EVAL_WEEK

logger = logging.getLogger(__name__)

CLICKHOUSE_URL = "https://play.clickhouse.com"
QUERY_TIMEOUT_S = 300
MIN_COMMENTS_PER_PAIR = 10
REVIEW_EVENT_TYPES = (
    "PullRequestReviewCommentEvent",
    "PullRequestReviewEvent",
    "IssueCommentEvent",
)
OUTPUT_PATH = DATA_DIR / "wild_reviews.json"


def _sql_str_list(items: list[str] | tuple[str, ...]) -> str:
    """Format a list of strings as a SQL `IN (...)` operand."""
    return ", ".join(f"'{item}'" for item in items)


def _clickhouse_query(sql: str, timeout: int = 120) -> dict[str, Any]:
    """POST a SQL query to the ClickHouse playground; return the parsed JSON response."""
    if "FORMAT" not in sql.upper():
        sql = sql.rstrip().rstrip(";") + "\nFORMAT JSONCompact"
    req = Request(f"{CLICKHOUSE_URL}/?user=play", data=sql.encode("utf-8"), method="POST")
    req.add_header("Content-Type", "text/plain")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (HTTPError, URLError, TimeoutError) as exc:
        logger.error("ClickHouse query failed: %s", exc)
        raise


def _build_query() -> str:
    since, until = EVAL_WEEK
    return f"""
    SELECT
        repo_name, number AS pr_number, title AS pr_title,
        actor_login AS bot, comment_id,
        body AS comment_body, path AS comment_path,
        diff_hunk AS comment_diff_hunk, position AS comment_position,
        created_at AS comment_created_at, review_state, event_type
    FROM github_events
    WHERE event_type IN ({_sql_str_list(REVIEW_EVENT_TYPES)})
      AND actor_login IN ({_sql_str_list(list(EVAL_BOTS.keys()))})
      AND repo_name IN ({_sql_str_list(list(EVAL_REPOS))})
      AND created_at >= '{since}' AND created_at < '{until}'
      AND number > 0
    ORDER BY repo_name, number, created_at
    """


def _row_to_comment(row: list[Any]) -> dict[str, Any]:
    return {
        "repo": row[0],
        "pr_number": int(row[1]),
        "pr_title": row[2] or "",
        "tool": row[3],
        "comment_id": int(row[4]) if row[4] else 0,
        "comment_body": row[5] or "",
        "comment_path": row[6] or "",
        "comment_diff_hunk": row[7] or "",
        "comment_position": int(row[8]) if row[8] else 0,
        "comment_created_at": str(row[9]),
        "review_state": row[10] if isinstance(row[10], str) else "",
        "event_type": row[11] if isinstance(row[11], str) else "",
    }


def _filter_thin_pairs(comments: list[dict[str, Any]], min_per_pair: int) -> list[dict[str, Any]]:
    """Drop (repo, bot) pairs with fewer than `min_per_pair` comments."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for c in comments:
        counts[(c["repo"], c["tool"])] += 1
    thin = {pair for pair, n in counts.items() if n < min_per_pair}
    if not thin:
        return comments

    before = len(comments)
    kept = [c for c in comments if (c["repo"], c["tool"]) not in thin]
    removed = before - len(kept)
    logger.info(
        "Dropped %d pairs with < %d comments (%d removed)", len(thin), min_per_pair, removed
    )
    return kept


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    since, _ = EVAL_WEEK
    logger.info(
        "Collecting reviews for %d bots, %d repos from %s...",
        len(EVAL_BOTS),
        len(EVAL_REPOS),
        since,
    )

    response = _clickhouse_query(_build_query(), timeout=QUERY_TIMEOUT_S)

    valid_pairs = {(bot, repo) for bot, repos in EVAL_BOTS.items() for repo in repos}
    comments = [
        _row_to_comment(row) for row in response["data"] if (row[3], row[0]) in valid_pairs
    ]

    if not comments:
        logger.error("No comments collected.")
        sys.exit(1)

    comments = _filter_thin_pairs(comments, MIN_COMMENTS_PER_PAIR)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(comments, f, indent=2, default=str)

    tool_counts: dict[str, int] = defaultdict(int)
    for c in comments:
        tool_counts[c["tool"]] += 1

    repo_count = len({c["repo"] for c in comments})
    logger.info("\n%d comments across %d repos", len(comments), repo_count)
    for bot, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        logger.info("  %-55s %5d", bot, count)
    logger.info("\nSaved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
