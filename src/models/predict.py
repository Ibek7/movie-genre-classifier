"""Inference utilities for movie-genre classification.

Exposes four public helpers:

* :func:`predict` — batch predictions from raw plot strings.
* :func:`predict_genre` — convenience wrapper for a single plot.
* :func:`predict_proba` — per-class probability scores for a batch.
* :func:`predict_with_confidence` — top genre + confidence score per plot.
* :func:`predict_top_k` — top-k genres with probabilities per plot.
* :func:`predict_from_csv` — read a CSV, predict, and write results back.
* :func:`batch_predict_from_dir` — run CSV inference on an entire directory.

A CLI entry-point (``mgc-predict``) is also provided via :func:`main`.
"""

import argparse
import joblib
from typing import List, Union
from pathlib import Path

from src.preprocessing.cleaner import normalize_text
from src.features.vectorizer import transform_plots
import pandas as pd


def load_vectorizer(vec_path: Union[str, Path]):
    """Load a fitted TF-IDF vectorizer from disk."""
    return joblib.load(vec_path)


def load_model(model_path: Union[str, Path]):
    """Load a trained sklearn model (NB, LR, etc.) from disk."""
    return joblib.load(model_path)


def preprocess_plots(plots: List[str]) -> List[str]:
    """Apply the same normalization you used in cleaning."""
    return [normalize_text(p) for p in plots]


def validate_artifact_paths(
    vec_path: Union[str, Path],
    model_path: Union[str, Path],
) -> tuple[Path, Path]:
    """Validate and return vectorizer/model paths as ``Path`` objects."""
    vec = Path(vec_path)
    model = Path(model_path)

    if not vec.exists():
        raise FileNotFoundError(f"Vectorizer not found: {vec_path}")
    if not model.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return vec, model


def top_k_from_proba_dict(proba: dict[str, float], k: int = 3) -> list[dict]:
    """Return top-*k* classes from a probability dictionary.

    Parameters
    ----------
    proba:
        Mapping of class label to probability.
    k:
        Number of ranked classes to return.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    ranked = sorted(proba.items(), key=lambda x: x[1], reverse=True)
    return [{"genre": genre, "confidence": round(score, 4)} for genre, score in ranked[:k]]


def predict(
    plots: List[str],
    vec_path: Union[str, Path],
    model_path: Union[str, Path]
) -> List[str]:
    """
    Given raw plot summaries, return predicted genres.
    - Normalizes text
    - Vectorizes using the saved TF-IDF
    - Runs model.predict
    """
    if not plots:
        raise ValueError("plots must contain at least one plot summary")

    cleaned = preprocess_plots(plots)
    if not any(cleaned):
        raise ValueError("plots must contain at least one non-empty summary")

    validated_vec_path, validated_model_path = validate_artifact_paths(vec_path, model_path)

    vec = load_vectorizer(validated_vec_path)
    model = load_model(validated_model_path)

    X = transform_plots(vec, pd.Series(cleaned))
    return model.predict(X).tolist()

def predict_genre(
    plot: str,
    vec_path: Union[str, Path],
    model_path: Union[str, Path]
) -> str:
    """
    Wrapper around predict() for a single plot.
    """
    return predict([plot], vec_path, model_path)[0]


def predict_proba(
    plots: List[str],
    vec_path: Union[str, Path],
    model_path: Union[str, Path],
) -> List[dict]:
    """Return per-class probabilities for each plot in *plots*.

    Requires the loaded model to expose ``predict_proba``
    (e.g. :class:`~sklearn.linear_model.LogisticRegression`).
    Multinomial Naive Bayes also supports this; plain SVMs do not.

    Parameters
    ----------
    plots:
        Raw plot strings to classify.
    vec_path:
        Path to the saved TF-IDF vectorizer.
    model_path:
        Path to the saved trained model.

    Returns
    -------
    list[dict]
        One dict per plot mapping class label → probability (float).

    Raises
    ------
    AttributeError
        If the model does not support ``predict_proba``.
    """
    if not plots:
        raise ValueError("plots must contain at least one plot summary")

    cleaned = preprocess_plots(plots)
    validated_vec_path, validated_model_path = validate_artifact_paths(vec_path, model_path)

    vec = load_vectorizer(validated_vec_path)
    model = load_model(validated_model_path)

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"{type(model).__name__} does not support predict_proba. "
            "Use a probabilistic model such as LogisticRegression or MultinomialNB."
        )

    X = transform_plots(vec, pd.Series(cleaned))
    proba_matrix = model.predict_proba(X)
    classes = list(model.classes_)
    return [dict(zip(classes, row.tolist())) for row in proba_matrix]


def predict_with_confidence(
    plots: List[str],
    vec_path: Union[str, Path],
    model_path: Union[str, Path],
) -> List[dict]:
    """Return each plot's top predicted genre together with its confidence score.

    This is a convenience wrapper around :func:`predict_proba` that surfaces
    only the winning class and its probability, making it easy to build
    downstream UIs or threshold-based rejection logic.

    Parameters
    ----------
    plots:
        Raw plot strings to classify.
    vec_path:
        Path to the saved TF-IDF vectorizer.
    model_path:
        Path to the saved trained model (must support ``predict_proba``).

    Returns
    -------
    list[dict]
        A list with one dict per plot containing keys ``"genre"`` (str) and
        ``"confidence"`` (float in ``[0, 1]``).

    Examples
    --------
    >>> results = predict_with_confidence(["A hero saves the world"], vec, model)
    >>> results[0].keys()
    dict_keys(['genre', 'confidence'])
    """
    proba_dicts = predict_proba(plots, vec_path, model_path)
    results = []
    for proba in proba_dicts:
        top_genre = max(proba, key=proba.__getitem__)
        results.append({"genre": top_genre, "confidence": round(proba[top_genre], 4)})
    return results


def predict_top_k(
    plots: List[str],
    vec_path: Union[str, Path],
    model_path: Union[str, Path],
    k: int = 3,
) -> List[List[dict]]:
    """Return the top-*k* most likely genres and their probabilities for each plot.

    Useful for building multi-label suggestions or ranking interfaces where the
    single best guess isn't sufficient.

    Parameters
    ----------
    plots:
        Raw plot strings to classify.
    vec_path:
        Path to the saved TF-IDF vectorizer.
    model_path:
        Path to the saved trained model (must support ``predict_proba``).
    k:
        Number of top genres to return per plot.  Clamped to the number of
        available classes if *k* exceeds it.

    Returns
    -------
    list[list[dict]]
        One list per plot, each containing up to *k* dicts of
        ``{"genre": str, "confidence": float}`` sorted by descending confidence.

    Raises
    ------
    ValueError
        If *k* ≤ 0.

    Examples
    --------
    >>> top = predict_top_k(["A hero saves the world"], vec, model, k=3)
    >>> top[0][0]["genre"]  # most likely genre
    'Action'
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    proba_dicts = predict_proba(plots, vec_path, model_path)
    return [top_k_from_proba_dict(proba, k=k) for proba in proba_dicts]


def predict_from_csv(
    input_csv: Union[str, Path],
    output_csv: Union[str, Path],
    vec_path: Union[str, Path],
    model_path: Union[str, Path]
):
    """
    Read `input_csv` (must have a 'Plot' column), predict genres,
    and write a new CSV with an added 'Predicted_Genre' column.
    """
    if not Path(input_csv).exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "Plot" not in df:
        raise ValueError("Input CSV must contain a 'Plot' column")
    if df.empty:
        raise ValueError("Input CSV is empty")

    plots = df["Plot"].fillna("").astype(str).tolist()
    df["Predicted_Genre"] = predict(plots, vec_path, model_path)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


def batch_predict_from_dir(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    vec_path: Union[str, Path],
    model_path: Union[str, Path],
    glob: str = "*.csv",
) -> List[Path]:
    """Run :func:`predict_from_csv` on every CSV in *input_dir*.

    This is useful for processing a folder of new movie batches without
    manually looping over files.

    Parameters
    ----------
    input_dir:
        Directory containing one or more CSV files with a ``Plot`` column.
    output_dir:
        Directory where annotated output CSVs will be written.
        Each output file has the same name as its input file.
    vec_path:
        Path to the fitted TF-IDF vectorizer.
    model_path:
        Path to the trained model.
    glob:
        File pattern used to discover CSVs inside *input_dir* (default ``*.csv``).

    Returns
    -------
    list[pathlib.Path]
        Paths of the output CSV files that were written.

    Raises
    ------
    FileNotFoundError
        If *input_dir* does not exist.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for csv_file in sorted(input_dir.glob(glob)):
        out_file = output_dir / csv_file.name
        predict_from_csv(csv_file, out_file, vec_path, model_path)
        written.append(out_file)

    print(f"Processed {len(written)} file(s) → {output_dir}")
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Predict movie genres from plot summaries"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Single‐text mode
    p1 = sub.add_parser("text", help="Predict genre for one or more plots")
    p1.add_argument(
        "--plots", nargs="+", required=True,
        help="Raw plot strings to classify"
    )
    p1.add_argument(
        "--vectorizer", required=True,
        help="Path to your TF-IDF vectorizer (.joblib)"
    )
    p1.add_argument(
        "--model", required=True,
        help="Path to your trained model (.joblib)"
    )

    # CSV‐batch mode
    p2 = sub.add_parser("csv", help="Predict genres from a CSV of plots")
    p2.add_argument(
        "--input-csv", required=True,
        help="CSV file with a 'Plot' column"
    )
    p2.add_argument(
        "--output-csv", required=True,
        help="Where to write the predictions"
    )
    p2.add_argument(
        "--vectorizer", required=True,
        help="Path to your TF-IDF vectorizer (.joblib)"
    )
    p2.add_argument(
        "--model", required=True,
        help="Path to your trained model (.joblib)"
    )

    args = parser.parse_args()

    if args.cmd == "text":
        preds = predict(args.plots, args.vectorizer, args.model)
        for plot, genre in zip(args.plots, preds):
            print(f"> Plot: {plot}\n→ Predicted genre: {genre}\n")
    else:  # args.cmd == "csv"
        predict_from_csv(args.input_csv, args.output_csv, args.vectorizer, args.model)


if __name__ == "__main__":
    main()