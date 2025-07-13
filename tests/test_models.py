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
    
    X_test, y_test = train_and_save_models(
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
    assert len(X_test) == len(y_test) == 1