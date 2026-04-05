"""Tests for src.utils.helpers shared utilities."""

import json
import pytest

from src.utils.helpers import ensure_dir, load_json, save_json, format_accuracy


# ---------------------------------------------------------------------------
# ensure_dir
# ---------------------------------------------------------------------------

def test_ensure_dir_creates_nested_directories(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert target.exists()
    assert target.is_dir()
    assert result == target


def test_ensure_dir_is_idempotent(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    ensure_dir(target)  # should not raise
    assert target.exists()


def test_ensure_dir_accepts_string_path(tmp_path):
    target = str(tmp_path / "string_path")
    result = ensure_dir(target)
    assert result.exists()


# ---------------------------------------------------------------------------
# save_json / load_json
# ---------------------------------------------------------------------------

def test_save_json_writes_valid_json(tmp_path):
    data = {"model": "lr", "accuracy": 0.72}
    path = tmp_path / "out.json"
    save_json(data, path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == data


def test_save_json_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "results.json"
    save_json({"ok": True}, path)
    assert path.exists()


def test_load_json_returns_dict(tmp_path):
    data = {"key": "value", "num": 42}
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))
    assert load_json(path) == data


def test_load_json_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_json(tmp_path / "missing.json")


def test_save_and_load_roundtrip(tmp_path):
    original = {"accuracy": 0.812, "classes": ["Action", "Drama"], "ready": True}
    path = tmp_path / "roundtrip.json"
    save_json(original, path)
    assert load_json(path) == original


# ---------------------------------------------------------------------------
# format_accuracy
# ---------------------------------------------------------------------------

def test_format_accuracy_default_decimals():
    assert format_accuracy(0.724) == "72.40%"


def test_format_accuracy_custom_decimals():
    assert format_accuracy(0.5, decimals=0) == "50%"
    assert format_accuracy(0.5, decimals=1) == "50.0%"


def test_format_accuracy_zero():
    assert format_accuracy(0.0) == "0.00%"


def test_format_accuracy_one():
    assert format_accuracy(1.0) == "100.00%"
