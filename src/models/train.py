"""Training pipeline for movie-genre classification models.

Loads a cleaned CSV, consolidates rare genres, fits a TF-IDF vectorizer,
trains Naive Bayes and Logistic Regression classifiers, evaluates them on a
held-out test set, and persists all artefacts to disk.
"""

import logging
import time
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

def train_and_save_models(
    data_path: str,
    vec_path: str,
    model_paths: dict[str, str],
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 5000,
    min_genre_samples: int = 100,
) -> tuple[Any, pd.Series, dict[str, Any]]:
    """Train NB and LR classifiers and persist all artefacts to disk.

    Parameters
    ----------
    data_path:
        Path to the cleaned CSV produced by :func:`src.preprocessing.cleaner.clean_and_save`.
    vec_path:
        Destination path for the fitted TF-IDF vectorizer (``.joblib``).
    model_paths:
        Mapping of ``{"nb": <path>, "lr": <path>}`` for the two model files.
    test_size:
        Fraction of data held out for evaluation (must be in ``(0, 1)``).
    random_state:
        Seed used for the train/test split to ensure reproducibility.
    max_features:
        Maximum vocabulary size passed to :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
    min_genre_samples:
        Genres with fewer than this many samples are merged into ``"other"``.

    Returns
    -------
    X_test : scipy sparse matrix
        TF-IDF feature matrix for the test split.
    y_test : pandas.Series
        True genre labels for the test split.
    performance_summary : dict
        Nested dict with experiment config, data stats, and per-model metrics.
    """
    # 1) Load data
    df = pd.read_csv(data_path)
    plots = df["Plot"]
    genres = df["Genre"].str.split("|")
    y = genres.apply(lambda g: g[0])
    
    # 2) OPTIMIZATION: Genre consolidation (from notebook)
    log.info("Original genres: %d", len(y.value_counts()))
    genre_counts = y.value_counts()
    
    # Keep genres with min_genre_samples+ OR top 15 most common
    common_genres = genre_counts[genre_counts >= min_genre_samples].index
    top_genres = genre_counts.head(15).index
    
    if len(common_genres) <= 15:
        chosen_genres = common_genres
        method = f"genres with {min_genre_samples}+ samples"
    else:
        chosen_genres = top_genres
        method = "top 15 genres"
    
    # Consolidate rare genres into 'other'
    y = y.where(y.isin(chosen_genres), other="other")
    
    log.info("Using %s: %d classes", method, len(y.value_counts()))
    log.info("Reduced genres: %s", list(y.value_counts().index))
    
    # 3) Split (use stratify only when possible)
    class_counts = y.value_counts()
    stratify_arg = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        plots, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )

    # Diagnostics
    log.debug("NaN count in training plots: %d", X_train.isnull().sum())
    log.debug("Average plot length:\n%s", X_train.str.len().describe())

    # 4) Fit TF-IDF vectorizer (matching successful notebook approach)
    log.info("Fitting TF-IDF vectorizer (max_features=%d) ...", max_features)
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tr = vectorizer.fit_transform(X_train)
    X_te = vectorizer.transform(X_test)
    
    log.info("TF-IDF features: %d  |  training samples: %d  |  classes: %d",
             X_tr.shape[1], X_tr.shape[0], len(set(y_train)))
    
    # Save vectorizer
    Path(vec_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, vec_path)
    log.info("Vectorizer saved to: %s", vec_path)
    
    # 5) Train models with timing
    log.info("Training Naive Bayes ...")
    start_time = time.time()
    nb = MultinomialNB()
    nb.fit(X_tr, y_train)
    nb_time = time.time() - start_time
    log.info("Naive Bayes trained in %.2f s", nb_time)
    Path(model_paths["nb"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(nb, model_paths["nb"])
    
    log.info("Training Logistic Regression ...")
    start_time = time.time()
    if y_train.nunique() < 2:
        single_class = y_train.iloc[0]
        log.warning(
            "Only one class in training data after consolidation; "
            "using constant fallback classifier."
        )
        lr = DummyClassifier(strategy="constant", constant=single_class)
        lr.fit(X_tr, y_train)
    else:
        lr = LogisticRegression(
            max_iter=500,
            random_state=random_state,
            C=1.0
        )
        lr.fit(X_tr, y_train)
        log.info("Logistic Regression converged in %s iterations", lr.n_iter_)

    lr_time = time.time() - start_time
    log.info("Logistic Regression trained in %.2f s", lr_time)
    Path(model_paths["lr"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr, model_paths["lr"])
    
    # 6) Evaluate models on test set
    log.info("--- Model Performance ---")
    y_pred_nb = nb.predict(X_te)
    y_pred_lr = lr.predict(X_te)
    
    nb_accuracy = accuracy_score(y_test, y_pred_nb)
    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    
    log.info("Naive Bayes accuracy: %.3f", nb_accuracy)
    log.info("Logistic Regression accuracy: %.3f", lr_accuracy)
    
    # 7) Save comprehensive performance summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    performance_summary = {
        'timestamp': timestamp,
        'experiment_config': {
            'max_features': max_features,
            'min_genre_samples': min_genre_samples,
            'test_size': test_size,
            'random_state': random_state
        },
        'data_stats': {
            'total_samples': len(df),
            'features_extracted': X_tr.shape[1],
            'classes_after_consolidation': len(set(y_train)),
            'train_samples': len(y_train),
            'test_samples': len(y_test)
        },
        'model_performance': {
            'naive_bayes': {
                'accuracy': float(nb_accuracy),
                'training_time_seconds': float(nb_time)
            },
            'logistic_regression': {
                'accuracy': float(lr_accuracy),
                'training_time_seconds': float(lr_time),
                'iterations': int(lr.n_iter_[0]) if hasattr(lr, 'n_iter_') else None
            }
        },
        'production_ready': bool(max(lr_accuracy, nb_accuracy) > 0.35),  # More realistic threshold
        'recommended_model': 'logistic_regression' if lr_accuracy > nb_accuracy else 'naive_bayes'
    }
    
    # Save performance summary
    summary_path = Path(vec_path).parent / f"performance_summary_{timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(performance_summary, f, indent=2)
    
    log.info("Performance summary saved to: %s", summary_path)
    log.info("Production ready: %s", performance_summary['production_ready'])
    log.info("Recommended model: %s", performance_summary['recommended_model'])
    
    return X_te, y_test, performance_summary