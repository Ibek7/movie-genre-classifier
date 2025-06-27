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