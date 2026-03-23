import pandas as pd
from src.preprocessing.cleaner import normalize_text, drop_duplicates, drop_missing, clean_and_save
from src.preprocessing.tokenizer import tokenize

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