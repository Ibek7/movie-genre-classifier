"""Tests for src.utils.helpers shared utilities."""

import json
import pytest
import pandas as pd

from src.utils.helpers import (
    ensure_dir,
    load_json,
    save_json,
    format_accuracy,
    compute_classification_report,
    get_top_genres,
    safe_divide,
    clamp_probability,
    flatten_genre_labels,
    build_label_index,
)


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


# ---------------------------------------------------------------------------
# compute_classification_report
# ---------------------------------------------------------------------------

def test_classification_report_returns_dict():
    report = compute_classification_report(["Action", "Drama"], ["Action", "Drama"])
    assert isinstance(report, dict)


def test_classification_report_contains_per_class_keys():
    report = compute_classification_report(
        ["Action", "Drama", "Action"],
        ["Action", "Action", "Action"],
    )
    assert "Action" in report
    assert "Drama" in report


def test_classification_report_contains_summary_keys():
    report = compute_classification_report(["Action", "Drama"], ["Action", "Drama"])
    assert "macro avg" in report
    assert "weighted avg" in report


def test_classification_report_perfect_precision():
    y = ["Comedy"] * 5
    report = compute_classification_report(y, y)
    assert report["Comedy"]["precision"] == 1.0
    assert report["Comedy"]["recall"] == 1.0


def test_classification_report_string_output():
    report_str = compute_classification_report(
        ["Action", "Drama"], ["Action", "Drama"], output_dict=False
    )
    assert isinstance(report_str, str)
    assert "Action" in report_str


# ---------------------------------------------------------------------------
# get_top_genres
# ---------------------------------------------------------------------------

def test_get_top_genres_returns_list():
    s = pd.Series(["Action|Drama", "Drama|Comedy", "Action"])
    result = get_top_genres(s, top_n=2)
    assert isinstance(result, list)


def test_get_top_genres_correct_order():
    s = pd.Series(["Drama", "Drama", "Action", "Comedy"])
    result = get_top_genres(s, top_n=3)
    assert result[0] == "Drama"  # most frequent


def test_get_top_genres_respects_top_n():
    s = pd.Series(["Drama", "Action", "Comedy", "Horror", "Thriller"])
    result = get_top_genres(s, top_n=3)
    assert len(result) <= 3


def test_get_top_genres_primary_only():
    s = pd.Series(["Action|Drama", "Action|Comedy", "Drama"])
    result = get_top_genres(s, top_n=5, primary_only=True)
    # primary label is before the pipe; Action appears twice
    assert result[0] == "Action"


def test_get_top_genres_all_labels():
    s = pd.Series(["Action|Drama", "Action|Drama"])
    result = get_top_genres(s, top_n=5, primary_only=False)
    # Both Action and Drama should appear when counting all labels
    assert "Action" in result
    assert "Drama" in result


# ---------------------------------------------------------------------------
# safe_divide
# ---------------------------------------------------------------------------

def test_safe_divide_regular_case():
    assert safe_divide(10, 2) == 5


def test_safe_divide_zero_denominator_returns_default():
    assert safe_divide(10, 0) == 0.0


def test_safe_divide_zero_denominator_custom_default():
    assert safe_divide(10, 0, default=-1) == -1


# ---------------------------------------------------------------------------
# clamp_probability
# ---------------------------------------------------------------------------

def test_clamp_probability_keeps_values_inside_range():
    assert clamp_probability(0.42) == 0.42


def test_clamp_probability_caps_high_values_to_one():
    assert clamp_probability(1.8) == 1.0


def test_clamp_probability_raises_low_values_to_zero():
    assert clamp_probability(-0.2) == 0.0


# ---------------------------------------------------------------------------
# flatten_genre_labels
# ---------------------------------------------------------------------------

def test_flatten_genre_labels_splits_and_flattens_pipe_delimited_values():
    s = pd.Series(["Action|Drama", "Comedy|Romance"])
    assert flatten_genre_labels(s) == ["Action", "Drama", "Comedy", "Romance"]


def test_flatten_genre_labels_strips_whitespace_and_drops_empty_tokens():
    s = pd.Series(["Action | Drama | ", "  Comedy  "])
    assert flatten_genre_labels(s) == ["Action", "Drama", "Comedy"]


def test_flatten_genre_labels_ignores_null_values():
    s = pd.Series([None, "Thriller|Crime"])
    assert flatten_genre_labels(s) == ["Thriller", "Crime"]


# ---------------------------------------------------------------------------
# build_label_index
# ---------------------------------------------------------------------------

def test_build_label_index_returns_sorted_mapping():
    result = build_label_index(["Drama", "Action", "Comedy"])
    assert result == {"Action": 0, "Comedy": 1, "Drama": 2}


def test_build_label_index_deduplicates_labels():
    result = build_label_index(["Action", "Drama", "Action"])
    assert len(result) == 2
    assert result["Action"] < result["Drama"]


def test_build_label_index_single_label():
    assert build_label_index(["Horror"]) == {"Horror": 0}
