import pandas as pd
from src.preprocessing.cleaner import normalize_text, drop_duplicates, drop_missing
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