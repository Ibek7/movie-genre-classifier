import pandas as pd
import pytest
from src.preprocessing.cleaner import (
    normalize_text,
    drop_duplicates,
    drop_missing,
    clean_and_save,
    word_count,
    sentence_count,
    truncate_plot,
    filter_short_plots,
)
from src.preprocessing.tokenizer import tokenize, tokenize_batch

def test_normalize_text():
    raw = "<p>Hello, WORLD!!!</p>\nNew   line."
    out = normalize_text(raw)
    assert "hello world new line" in out

def test_drop_duplicates(tmp_path):
    df = pd.DataFrame({
        "Title": ["A", "A"],
        "Plot": ["same", "same"],
        "Genre": ["X", "X"]
    })
    deduped = drop_duplicates(df)
    assert len(deduped) == 1

def test_drop_missing(tmp_path):
    df = pd.DataFrame({
        "Title": ["A", "B"],
        "Plot": ["ok", None],
        "Genre": ["X", "Y"]
    })
    cleaned = drop_missing(df)
    assert len(cleaned) == 1 and cleaned.iloc[0]["Title"] == "A"

def test_tokenize_basic():
    tokens = tokenize("The QUICK brown fox.")
    assert "quick" in tokens and "fox" in tokens and "the" not in tokens


def test_normalize_text_handles_none():
    assert normalize_text(None) == ""


def test_normalize_text_handles_non_string():
    assert normalize_text(12345) == "12345"


def test_clean_and_save_creates_output_directory(tmp_path):
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "nested" / "processed" / "cleaned.csv"

    pd.DataFrame(
        {
            "Title": ["A"],
            "Plot": ["<p>Hello</p>"],
            "Genre": ["Drama"],
        }
    ).to_csv(input_csv, index=False)

    clean_and_save(str(input_csv), str(output_csv))

    assert output_csv.exists()


# ---------------------------------------------------------------------------
# tokenize_batch tests
# ---------------------------------------------------------------------------

def test_tokenize_batch_returns_one_list_per_input():
    texts = ["The quick brown fox", "A brave new world"]
    result = tokenize_batch(texts)
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)


def test_tokenize_batch_removes_stop_words_by_default():
    result = tokenize_batch(["The QUICK brown fox"])
    tokens = result[0]
    assert "the" not in tokens
    assert "quick" in tokens


def test_tokenize_batch_keeps_stop_words_when_disabled():
    result = tokenize_batch(["The quick brown fox"], remove_stopwords=False)
    tokens = result[0]
    assert "the" in tokens


def test_tokenize_batch_empty_corpus_returns_empty_list():
    assert tokenize_batch([]) == []


def test_tokenize_batch_order_matches_input():
    texts = ["action hero", "romantic comedy drama"]
    result = tokenize_batch(texts)
    # 'action' appears in first doc, not second
    assert any("action" in t for t in result[0])
    assert all("action" not in t for t in result[1])


# ---------------------------------------------------------------------------
# word_count tests
# ---------------------------------------------------------------------------

def test_word_count_basic():
    assert word_count("A hero saves the world") == 5


def test_word_count_single_word():
    assert word_count("Drama") == 1


def test_word_count_empty_string():
    assert word_count("") == 0


def test_word_count_none_input():
    assert word_count(None) == 0


def test_word_count_extra_spaces():
    assert word_count("  hello   world  ") == 2


# ---------------------------------------------------------------------------
# sentence_count tests
# ---------------------------------------------------------------------------

def test_sentence_count_basic():
    assert sentence_count("He ran. She laughed! Why?") == 3


def test_sentence_count_single_sentence():
    assert sentence_count("A boy loved movies.") == 1


def test_sentence_count_no_punctuation():
    assert sentence_count("no punctuation here") == 1


def test_sentence_count_empty_string():
    assert sentence_count("") == 0


def test_sentence_count_none_input():
    assert sentence_count(None) == 0


# ---------------------------------------------------------------------------
# truncate_plot tests
# ---------------------------------------------------------------------------

def test_truncate_plot_trims_to_max_words():
    text = "one two three four five six seven eight nine ten"
    assert truncate_plot(text, max_words=5) == "one two three four five"


def test_truncate_plot_unchanged_when_shorter_than_limit():
    text = "short plot"
    assert truncate_plot(text, max_words=100) == "short plot"


def test_truncate_plot_empty_string_returns_empty():
    assert truncate_plot("", max_words=10) == ""


def test_truncate_plot_none_returns_empty():
    assert truncate_plot(None, max_words=10) == ""


def test_truncate_plot_zero_max_words_raises():
    with pytest.raises(ValueError, match="max_words"):
        truncate_plot("some text", max_words=0)


def test_truncate_plot_negative_max_words_raises():
    with pytest.raises(ValueError, match="max_words"):
        truncate_plot("some text", max_words=-5)


# ---------------------------------------------------------------------------
# filter_short_plots tests
# ---------------------------------------------------------------------------

def test_filter_short_plots_removes_short_rows():
    df = pd.DataFrame({"Plot": ["short", "a much longer plot with many more words in it here"]})
    result = filter_short_plots(df, min_words=5)
    assert len(result) == 1
    assert "longer" in result["Plot"].iloc[0]


def test_filter_short_plots_keeps_all_when_all_long_enough():
    df = pd.DataFrame({"Plot": ["this has five words here", "another five word plot now"]})
    result = filter_short_plots(df, min_words=5)
    assert len(result) == 2


def test_filter_short_plots_removes_all_when_all_short():
    df = pd.DataFrame({"Plot": ["tiny", "stub"]})
    result = filter_short_plots(df, min_words=10)
    assert len(result) == 0


def test_filter_short_plots_invalid_min_words_raises():
    df = pd.DataFrame({"Plot": ["some text"]})
    with pytest.raises(ValueError, match="min_words"):
        filter_short_plots(df, min_words=0)


def test_filter_short_plots_missing_column_raises():
    df = pd.DataFrame({"text": ["some text"]})
    with pytest.raises(ValueError, match="not found"):
        filter_short_plots(df, min_words=5)


def test_filter_short_plots_handles_null_plots():
    df = pd.DataFrame({"Plot": [None, "a plot with enough words to pass the filter"]})
    result = filter_short_plots(df, min_words=5)
    # None is treated as empty (0 words), so it should be filtered out
    assert len(result) == 1
