"""Pure helpers in collect_wild_reviews: SQL formatting, row parsing, filtering."""

from review_bot_comparison.collect_wild_reviews import (
    _filter_thin_pairs,
    _row_to_comment,
    _sql_str_list,
)

# ── _sql_str_list ────────────────────────────────────────────────────────────


def test_sql_str_list_quotes_each_item():
    assert _sql_str_list(["a", "b"]) == "'a', 'b'"


def test_sql_str_list_accepts_tuple():
    assert _sql_str_list(("x",)) == "'x'"


def test_sql_str_list_single_element():
    assert _sql_str_list(["only"]) == "'only'"


def test_sql_str_list_empty_returns_empty_string():
    assert _sql_str_list([]) == ""


# ── _row_to_comment ──────────────────────────────────────────────────────────

# Row schema: repo_name, pr_number, pr_title, actor_login, comment_id,
#             comment_body, comment_path, comment_diff_hunk, comment_position,
#             created_at, review_state, event_type


def _full_row():
    return [
        "owner/repo",
        42,
        "Add feature",
        "some-bot[bot]",
        9001,
        "looks good",
        "src/foo.py",
        "@@ -1 +1 @@",
        7,
        "2026-03-17T00:00:00Z",
        "APPROVED",
        "PullRequestReviewEvent",
    ]


def test_row_to_comment_full():
    comment = _row_to_comment(_full_row())
    assert comment == {
        "repo": "owner/repo",
        "pr_number": 42,
        "pr_title": "Add feature",
        "tool": "some-bot[bot]",
        "comment_id": 9001,
        "comment_body": "looks good",
        "comment_path": "src/foo.py",
        "comment_diff_hunk": "@@ -1 +1 @@",
        "comment_position": 7,
        "comment_created_at": "2026-03-17T00:00:00Z",
        "review_state": "APPROVED",
        "event_type": "PullRequestReviewEvent",
    }


def test_row_to_comment_falsy_strings_become_empty():
    row = _full_row()
    row[2] = None  # pr_title
    row[5] = None  # comment_body
    row[6] = None  # comment_path
    row[7] = None  # comment_diff_hunk
    comment = _row_to_comment(row)
    assert comment["pr_title"] == ""
    assert comment["comment_body"] == ""
    assert comment["comment_path"] == ""
    assert comment["comment_diff_hunk"] == ""


def test_row_to_comment_falsy_ints_become_zero():
    row = _full_row()
    row[4] = None  # comment_id
    row[8] = None  # comment_position
    comment = _row_to_comment(row)
    assert comment["comment_id"] == 0
    assert comment["comment_position"] == 0


def test_row_to_comment_non_string_review_state_becomes_empty():
    # ClickHouse can hand back nulls or other shapes; we only keep strings.
    row = _full_row()
    row[10] = None
    row[11] = None
    comment = _row_to_comment(row)
    assert comment["review_state"] == ""
    assert comment["event_type"] == ""


def test_row_to_comment_coerces_string_pr_number_to_int():
    row = _full_row()
    row[1] = "42"  # CH may return PR numbers as strings
    assert _row_to_comment(row)["pr_number"] == 42


# ── _filter_thin_pairs ───────────────────────────────────────────────────────


def _comment(repo: str, tool: str) -> dict:
    return {"repo": repo, "tool": tool}


def test_filter_thin_pairs_drops_below_threshold():
    comments = [_comment("r1", "bot")] * 5 + [_comment("r2", "bot")] * 20
    kept = _filter_thin_pairs(comments, min_per_pair=10)
    assert len(kept) == 20
    assert all(c["repo"] == "r2" for c in kept)


def test_filter_thin_pairs_keeps_pair_at_threshold():
    comments = [_comment("r", "bot")] * 10
    assert len(_filter_thin_pairs(comments, min_per_pair=10)) == 10


def test_filter_thin_pairs_no_thin_returns_input_unchanged():
    comments = [_comment("r", "bot")] * 100
    assert _filter_thin_pairs(comments, min_per_pair=10) is comments


def test_filter_thin_pairs_empty_input():
    assert _filter_thin_pairs([], min_per_pair=10) == []


def test_filter_thin_pairs_distinguishes_pair_keys():
    # Same repo, two tools — one thin, one thick — only the thick stays.
    comments = [_comment("r", "thin-bot")] * 3 + [_comment("r", "thick-bot")] * 50
    kept = _filter_thin_pairs(comments, min_per_pair=10)
    assert {c["tool"] for c in kept} == {"thick-bot"}
