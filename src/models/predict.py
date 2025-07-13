import argparse
import joblib
from typing import List, Union
from pathlib import Path

from preprocessing.cleaner import normalize_text
from features.vectorizer import transform_plots
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
    vec = load_vectorizer(vec_path)
    model = load_model(model_path)

    cleaned = preprocess_plots(plots)
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
    df = pd.read_csv(input_csv)
    if "Plot" not in df:
        raise ValueError("Input CSV must contain a 'Plot' column")
    df["Predicted_Genre"] = predict(df["Plot"].tolist(), vec_path, model_path)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


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