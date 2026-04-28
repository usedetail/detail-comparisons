"""LLM judge logic — particularly the position-bias swap."""

from unittest.mock import MagicMock

import anthropic

from review_bot_comparison.cache import DiskCache
from review_bot_comparison.llm import _judgment_cache_key, judge_pair


def _mock_client(winner: str) -> MagicMock:
    """Anthropic client mock that always returns `winner` as the chosen label."""
    client = MagicMock(spec=anthropic.Anthropic)
    tool_block = MagicMock(spec=anthropic.types.ToolUseBlock)
    tool_block.name = "select_winner"
    tool_block.input = {"winner": winner, "justification": "test reason"}
    response = MagicMock()
    response.content = [tool_block]
    client.messages.create.return_value = response
    return client


def _empty_cache(tmp_path) -> DiskCache:
    """Fresh on-disk cache scoped to the test's temp dir."""
    return DiskCache("judgments", directory=tmp_path)


def test_returns_a_or_b(tmp_path):
    client = _mock_client("a")
    cache = _empty_cache(tmp_path)
    result, justification = judge_pair(client, "finding 1", "finding 2", swap=False, cache=cache)
    assert result in ("a", "b")
    assert isinstance(justification, str)


def test_swap_flips_returned_label(tmp_path):
    """Model is biased to always pick first-presented. With swap=True, the
    'a'-vs-'b' mapping is reversed, so we should get 'b' back."""
    client = _mock_client("a")  # always picks first-presented

    no_swap_result, _ = judge_pair(client, "X1", "Y1", swap=False, cache=_empty_cache(tmp_path))
    assert no_swap_result == "a"

    swap_result, _ = judge_pair(client, "X2", "Y2", swap=True, cache=_empty_cache(tmp_path))
    assert swap_result == "b"


def test_cache_hit_skips_api_call(tmp_path):
    client = _mock_client("a")
    cache = _empty_cache(tmp_path)

    judge_pair(client, "X", "Y", swap=False, cache=cache)
    assert client.messages.create.call_count == 1

    judge_pair(client, "X", "Y", swap=False, cache=cache)
    assert client.messages.create.call_count == 1  # no new call

    # Order-independent: swapping inputs still hits cache.
    judge_pair(client, "Y", "X", swap=False, cache=cache)
    assert client.messages.create.call_count == 1


# ── _judgment_cache_key ──────────────────────────────────────────────────────


def test_cache_key_is_order_independent():
    assert _judgment_cache_key("foo", "bar") == _judgment_cache_key("bar", "foo")


def test_cache_key_differs_for_different_pairs():
    assert _judgment_cache_key("foo", "bar") != _judgment_cache_key("foo", "baz")


def test_cache_key_is_deterministic():
    assert _judgment_cache_key("foo", "bar") == _judgment_cache_key("foo", "bar")


def test_cache_key_same_pair_distinct_summaries():
    """Two distinct summaries that compare equal under sort still hash apart."""
    # "a" sorts before "b"; (a,b) and (a,a) must not collide.
    assert _judgment_cache_key("a", "b") != _judgment_cache_key("a", "a")
