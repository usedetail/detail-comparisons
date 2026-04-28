"""Sanity checks on the eval configuration."""

from review_bot_comparison.config import EVAL_BOTS, EVAL_REPOS, EVAL_WEEK


def test_each_bot_has_at_least_one_repo():
    for bot, repos in EVAL_BOTS.items():
        assert len(repos) >= 1, f"{bot} has no repos"


def test_no_duplicate_repos_per_bot():
    for bot, repos in EVAL_BOTS.items():
        assert len(repos) == len(set(repos)), f"{bot} has duplicate repos"


def test_eval_repos_is_derived_from_eval_bots():
    expected = sorted({repo for repos in EVAL_BOTS.values() for repo in repos})
    assert expected == EVAL_REPOS


def test_eval_week_is_inclusive_start_exclusive_end():
    start, end = EVAL_WEEK
    assert start < end, "EVAL_WEEK should have start < end"
