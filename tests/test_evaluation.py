import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from src.preprocessing.cleaner import clean_and_save
from src.models.train import train_and_save_models

def test_evaluation_pipeline(tmp_path):
    # 1) Create a toy dataset
    df = pd.DataFrame({
        "Plot": [
            "A hero saves the world", 
            "A love story with a tragic end",
            "Spaceships explore the galaxy",
            "A detective solves a mysterious crime"
        ],
        "Genre": [
            "Action|Adventure",
            "Romance|Drama",
            "Sci-Fi|Adventure",
            "Mystery|Thriller"
        ]
    })
    raw_file = tmp_path / "raw.csv"
    df.to_csv(raw_file, index=False)

    # 2) Clean the data
    cleaned_file = tmp_path / "cleaned.csv"
    clean_and_save(str(raw_file), str(cleaned_file))
    df_clean = pd.read_csv(cleaned_file)
    assert not df_clean.empty

    # 3) Train models on cleaned data
    vec_file = tmp_path / "vec.joblib"
    nb_file = tmp_path / "nb.joblib"
    lr_file = tmp_path / "lr.joblib"
    X_test, y_test = train_and_save_models(
        str(cleaned_file),
        str(vec_file),
        {"nb": str(nb_file), "lr": str(lr_file)},
        test_size=0.5,
        random_state=0
    )

    # 4) Load artifacts and evaluate
    vec = joblib.load(str(vec_file))
    nb_model = joblib.load(str(nb_file))
    lr_model = joblib.load(str(lr_file))
    X_te = vec.transform(X_test)

    # 5) Predictions and accuracy checks
    y_pred_nb = nb_model.predict(X_te)
    y_pred_lr = lr_model.predict(X_te)
    assert 0.0 <= accuracy_score(y_test, y_pred_nb) <= 1.0
    assert 0.0 <= accuracy_score(y_test, y_pred_lr) <= 1.0