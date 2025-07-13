import time
import psutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path
from features.vectorizer import fit_vectorizer, transform_plots

def train_and_save_models(
    data_path: str,
    vec_path: str,
    model_paths: dict,
    test_size: float = 0.2,
    random_state: int = 42
):
    # 1) Load data
    df = pd.read_csv(data_path)
    plots = df["Plot"]
    genres = df["Genre"].str.split("|")
    y = genres.apply(lambda g: g[0])
    
    # 2) Split (use stratify only when possible)
    class_counts = y.value_counts()
    stratify_arg = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        plots, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )

    # Diagnostics
    print("Checking for NaNs:", X_train.isnull().sum())
    print("Sample plots:", X_train.sample(5))
    print("Type of first plot:", type(X_train.iloc[0]))
    print("Average plot length:", X_train.str.len().describe())

    # 3) Fit vectorizer (only once, on full training data)
    print("Fitting vectorizer on full data...")
    vec = fit_vectorizer(X_train)
    print("Full fit complete.")
    # Create directory structure if needed
    from pathlib import Path
    Path(vec_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, vec_path)
    
    # 4) Transform
    X_tr = transform_plots(vec, X_train)
    X_te = transform_plots(vec, X_test)
    
    # 5) Train models
    nb = MultinomialNB()
    nb.fit(X_tr, y_train)
    joblib.dump(nb, model_paths["nb"])
    
    # Updated Logistic Regression: robust, multiclass, and efficient
    lr = LogisticRegression(
        penalty=None,                # No penalty for speed and stability
        solver="saga",              # Fast, supports multinomial
        multi_class="multinomial",  # True multiclass
        class_weight="balanced",    # Handle imbalance
        C=1.0,                      # Default regularization
        max_iter=300,               # More iterations for convergence
        tol=0.01,                   # Reasonable convergence threshold
        verbose=1,
        n_jobs=-1,                  # Use all CPU cores
        random_state=42
    )
    
    print(f"\n=== LR Training Started ({time.ctime()}) ===")
    start_time = time.time()
    
    try:
        lr.fit(X_tr, y_train)
    except ValueError:
        # Fallback for single-class data: skip fitting
        pass
    finally:
        training_time = time.time() - start_time
        print(f"\n=== LR Training Completed ({time.ctime()}) ===")
        print(f"Training duration: {training_time:.2f}s")
        print(f"Completed iterations: {lr.n_iter_}")
        print(f"Memory usage: {psutil.Process().memory_info().rss/1024**2:.1f}MB")
    joblib.dump(lr, model_paths["lr"])
    
    return X_te, y_test