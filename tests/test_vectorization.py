import pandas as pd
import pytest
from src.features.vectorizer import (
    fit_vectorizer,
    transform_plots,
    save_vectorizer,
    load_vectorizer,
    get_vocabulary_size,
)

def test_vectorizer_shapes():
    sample = pd.Series([
        "the quick brown fox",
        "jumps over the lazy dog"
    ])
    vec = fit_vectorizer(sample, max_features=10, ngram_range=(1,1))
    matrix = transform_plots(vec, sample)
    # Expect 2 rows and ≤ 10 columns
    assert matrix.shape[0] == 2
    assert matrix.shape[1] <= 10


def test_vectorizer_respects_stop_words_and_ngrams():
    sample = pd.Series([
        "the quick brown fox",
        "the quick blue fox"
    ])

    vec = fit_vectorizer(
        sample,
        max_features=50,
        ngram_range=(1, 2),
        min_df=1,
        stop_words="english",
    )

    vocab = set(vec.get_feature_names_out())
    assert "the" not in vocab
    assert "quick brown" in vocab or "quick blue" in vocab


def test_vectorizer_defaults_work_on_tiny_corpus():
    sample = pd.Series([
        "alpha beta",
        "beta gamma"
    ])

    vec = fit_vectorizer(sample)
    matrix = transform_plots(vec, sample)

    assert matrix.shape[0] == 2
    assert matrix.shape[1] > 0


# ---------------------------------------------------------------------------
# get_vocabulary_size
# ---------------------------------------------------------------------------

def test_get_vocabulary_size_returns_positive_int():
    sample = pd.Series(["action drama thriller", "comedy romance horror"])
    vec = fit_vectorizer(sample, max_features=50, min_df=1)
    size = get_vocabulary_size(vec)
    assert isinstance(size, int)
    assert size > 0


def test_get_vocabulary_size_respects_max_features():
    sample = pd.Series(["alpha beta gamma delta epsilon"] * 5)
    vec = fit_vectorizer(sample, max_features=3, min_df=1)
    assert get_vocabulary_size(vec) <= 3


def test_get_vocabulary_size_unfitted_raises():
    from sklearn.feature_extraction.text import TfidfVectorizer
    unfitted = TfidfVectorizer()
    with pytest.raises(ValueError, match="not been fitted"):
        get_vocabulary_size(unfitted)


# ---------------------------------------------------------------------------
# save_vectorizer / load_vectorizer roundtrip
# ---------------------------------------------------------------------------

def test_save_and_load_vectorizer_roundtrip(tmp_path):
    sample = pd.Series(["science fiction space", "romantic comedy drama"])
    vec = fit_vectorizer(sample, max_features=20, min_df=1)
    path = tmp_path / "vec.joblib"
    save_vectorizer(vec, path)
    loaded = load_vectorizer(path)
    # Vocabulary should be identical after roundtrip
    assert vec.vocabulary_ == loaded.vocabulary_


def test_load_vectorizer_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_vectorizer(tmp_path / "nonexistent.joblib")


def test_save_vectorizer_creates_parent_dirs(tmp_path):
    sample = pd.Series(["drama action", "comedy horror"])
    vec = fit_vectorizer(sample, max_features=10, min_df=1)
    nested = tmp_path / "a" / "b" / "c" / "vec.joblib"
    save_vectorizer(vec, nested)
    assert nested.exists()
