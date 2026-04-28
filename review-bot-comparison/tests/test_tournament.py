"""Tournament formatting helpers."""

from review_bot_comparison.tournament import format_bot_comment, format_detail_finding


class TestFormatBotComment:
    def test_full_comment_includes_all_parts(self):
        result = format_bot_comment(
            {
                "comment_path": "src/main.py",
                "comment_diff_hunk": "@@ -10,3 +10,5 @@",
                "comment_body": "Missing null check",
            }
        )
        assert "src/main.py" in result
        assert "@@ -10,3 +10,5 @@" in result
        assert "Missing null check" in result

    def test_body_only(self):
        assert format_bot_comment({"comment_body": "Looks good"}) == "Looks good"

    def test_empty_returns_empty_string(self):
        assert format_bot_comment({}) == ""

    def test_long_diff_hunk_is_truncated(self):
        result = format_bot_comment({"comment_diff_hunk": "x" * 1000, "comment_body": "note"})
        # 500-char hunk + newline + 4-char body = 505
        assert len(result) <= 505


class TestFormatDetailFinding:
    def test_full_finding_includes_all_parts(self):
        result = format_detail_finding(
            {
                "file_path": "lib/cache.py",
                "title": "Race condition in cache",
                "summary": "Two threads can write simultaneously",
            }
        )
        assert "lib/cache.py" in result
        assert "Race condition" in result
        assert "Two threads" in result

    def test_title_only(self):
        assert format_detail_finding({"title": "Bug found"}) == "Bug found"

    def test_empty_returns_empty_string(self):
        assert format_detail_finding({}) == ""
