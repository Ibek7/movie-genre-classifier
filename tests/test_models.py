import pandas as pd
from src.models.train import train_and_save_models
import os

def test_train_and_save(tmp_path):
    # Prepare a tiny toy dataset
    df = pd.DataFrame({
        "Plot": ["A brave hero fights dragons", "A shy teenager learns math"],
        "Genre": ["Action|Fantasy", "Drama|Education"]
    })
    data_file = tmp_path / "data.csv"
    df.to_csv(data_file, index=False)
    
    vec_file = tmp_path / "vec.joblib"
    nb_file = tmp_path / "nb.joblib"
    lr_file = tmp_path / "lr.joblib"
    
    X_test, y_test, summary = train_and_save_models(
        str(data_file),
        str(vec_file),
        {"nb": str(nb_file), "lr": str(lr_file)},
        test_size=0.5,
        random_state=0
    )
    
    # Check artefacts exist
    assert os.path.exists(vec_file)
    assert os.path.exists(nb_file)
    assert os.path.exists(lr_file)
    assert X_test.shape[0] == len(y_test) == 1
    assert "recommended_model" in summary


def test_train_and_save_handles_single_class_after_consolidation(tmp_path):
    df = pd.DataFrame({
        "Plot": [
            "A brave hero fights dragons",
            "A shy teenager learns math",
            "A detective solves a mystery"
        ],
        "Genre": ["Action|Fantasy", "Drama|Education", "Crime|Mystery"]
    })
    data_file = tmp_path / "single_class_data.csv"
    df.to_csv(data_file, index=False)

    vec_file = tmp_path / "vec.joblib"
    nb_file = tmp_path / "nb.joblib"
    lr_file = tmp_path / "lr.joblib"

    _, _, summary = train_and_save_models(
        str(data_file),
        str(vec_file),
        {"nb": str(nb_file), "lr": str(lr_file)},
        test_size=0.34,
        random_state=0,
        min_genre_samples=100,
    )

    assert os.path.exists(vec_file)
    assert os.path.exists(nb_file)
    assert os.path.exists(lr_file)
    assert summary["data_stats"]["classes_after_consolidation"] == 1