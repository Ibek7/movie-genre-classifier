"""movie-genre-classifier — root package marker.

The actual library code lives under :mod:`src`.  This file exists so that
tools such as ``pytest`` and ``setuptools`` can discover the project root
when the repository is installed in editable mode (``pip install -e .``).

Quick-start
-----------
Train::

    python -m src.main \\
        --data-path data/processed/cleaned_plots.csv \\
        --max-features 5000

Predict (single plot)::

    from src.models.predict import predict_genre
    genre = predict_genre(
        "A detective uncovers a conspiracy",
        vec_path="models/production_vectorizer.joblib",
        model_path="models/lr.joblib",
    )

Version information is maintained in :attr:`src.__version__`.
"""
