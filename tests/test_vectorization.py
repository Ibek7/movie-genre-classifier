import pandas as pd
from src.features.vectorizer import fit_vectorizer, transform_plots

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