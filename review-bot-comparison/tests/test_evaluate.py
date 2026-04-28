"""Pure helpers in the evaluate CLI."""

from review_bot_comparison.evaluate import _group_by_repo


def test_group_by_repo_partitions_bots_per_repo():
    wild = [
        {"repo": "r1", "tool": "bot-a", "comment_id": 1},
        {"repo": "r1", "tool": "bot-a", "comment_id": 2},
        {"repo": "r1", "tool": "bot-b", "comment_id": 3},
        {"repo": "r2", "tool": "bot-a", "comment_id": 4},
    ]
    bot_by_repo, _ = _group_by_repo(wild, [])
    assert sorted(bot_by_repo.keys()) == ["r1", "r2"]
    assert sorted(bot_by_repo["r1"].keys()) == ["bot-a", "bot-b"]
    assert len(bot_by_repo["r1"]["bot-a"]) == 2
    assert len(bot_by_repo["r1"]["bot-b"]) == 1
    assert len(bot_by_repo["r2"]["bot-a"]) == 1


def test_group_by_repo_groups_detail_findings():
    detail = [
        {"repo": "r1", "title": "f1"},
        {"repo": "r1", "title": "f2"},
        {"repo": "r2", "title": "f3"},
    ]
    _, detail_by_repo = _group_by_repo([], detail)
    assert sorted(detail_by_repo.keys()) == ["r1", "r2"]
    assert len(detail_by_repo["r1"]) == 2
    assert len(detail_by_repo["r2"]) == 1


def test_group_by_repo_empty_inputs_return_empty_groups():
    bot_by_repo, detail_by_repo = _group_by_repo([], [])
    assert dict(bot_by_repo) == {}
    assert dict(detail_by_repo) == {}


def test_group_by_repo_unknown_repo_lookup_returns_empty():
    bot_by_repo, _ = _group_by_repo([], [])
    # defaultdict semantics: lookup creates empty entry
    assert bot_by_repo.get("unseen", {}) == {}
