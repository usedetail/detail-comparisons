"""DiskCache and text_hash."""

import json

from review_bot_comparison.cache import DiskCache, text_hash

# ── text_hash ────────────────────────────────────────────────────────────────


def test_text_hash_is_16_chars():
    assert len(text_hash("anything")) == 16


def test_text_hash_is_deterministic():
    assert text_hash("hello") == text_hash("hello")


def test_text_hash_distinguishes_inputs():
    assert text_hash("hello") != text_hash("world")


def test_text_hash_handles_unicode():
    # Should not raise; should be stable.
    h1 = text_hash("café — 日本語")
    h2 = text_hash("café — 日本語")
    assert h1 == h2
    assert len(h1) == 16


def test_text_hash_distinguishes_empty_from_whitespace():
    assert text_hash("") != text_hash(" ")


# ── DiskCache ────────────────────────────────────────────────────────────────


def test_load_from_missing_file_is_empty(tmp_path):
    cache = DiskCache("nonexistent", directory=tmp_path)
    cache.load()
    assert len(cache) == 0


def test_set_and_get(tmp_path):
    cache = DiskCache("c", directory=tmp_path)
    cache.set("k", "v")
    assert cache.get("k") == "v"
    assert "k" in cache
    assert len(cache) == 1


def test_get_missing_key_returns_none(tmp_path):
    cache = DiskCache("c", directory=tmp_path)
    assert cache.get("nope") is None
    assert "nope" not in cache


def test_set_overwrites(tmp_path):
    cache = DiskCache("c", directory=tmp_path)
    cache.set("k", "v1")
    cache.set("k", "v2")
    assert cache.get("k") == "v2"
    assert len(cache) == 1


def test_clear_drops_all_entries(tmp_path):
    cache = DiskCache("c", directory=tmp_path)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()
    assert len(cache) == 0
    assert cache.get("a") is None


def test_save_then_load_round_trip(tmp_path):
    cache_a = DiskCache("c", directory=tmp_path)
    cache_a.set("k1", "v1")
    cache_a.set("k2", {"nested": [1, 2, 3]})
    cache_a.save()

    cache_b = DiskCache("c", directory=tmp_path)
    cache_b.load()
    assert cache_b.get("k1") == "v1"
    assert cache_b.get("k2") == {"nested": [1, 2, 3]}
    assert len(cache_b) == 2


def test_save_creates_parent_directory(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    cache = DiskCache("c", directory=nested)
    cache.set("k", "v")
    cache.save()
    assert nested.exists()
    assert cache.path.exists()


def test_save_writes_valid_json(tmp_path):
    cache = DiskCache("c", directory=tmp_path)
    cache.set("k", {"x": 1})
    cache.save()
    assert json.loads(cache.path.read_text()) == {"k": {"x": 1}}


def test_distinct_names_share_directory_not_data(tmp_path):
    a = DiskCache("a", directory=tmp_path)
    b = DiskCache("b", directory=tmp_path)
    a.set("k", "from_a")
    b.set("k", "from_b")
    assert a.get("k") == "from_a"
    assert b.get("k") == "from_b"
    a.save()
    b.save()
    assert (tmp_path / "a.json").exists()
    assert (tmp_path / "b.json").exists()
