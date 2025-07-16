import time
import psutil
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

def train_and_save_models(
    data_path: str,
    vec_path: str,
    model_paths: dict,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 5000,  # Add feature limit parameter
    min_genre_samples: int = 100  # Add genre consolidation parameter
):
    # 1) Load data
    df = pd.read_csv(data_path)
    plots = df["Plot"]
    genres = df["Genre"].str.split("|")
    y = genres.apply(lambda g: g[0])
    
    # 2) OPTIMIZATION: Genre consolidation (from notebook)
    print(f"Original genres: {len(y.value_counts())}")
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
    
    print(f"Using {method}: {len(y.value_counts())} classes")
    print(f"Reduced genres: {list(y.value_counts().index)}")
    
    # 3) Split (use stratify only when possible)
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

    # 4) Fit TF-IDF vectorizer (matching successful notebook approach)
    print("Fitting TF-IDF vectorizer with optimized features...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tr = vectorizer.fit_transform(X_train)
    X_te = vectorizer.transform(X_test)
    
    print(f"TF-IDF features: {X_tr.shape[1]}")
    print(f"Training samples: {X_tr.shape[0]}")
    print(f"Unique classes: {len(set(y_train))}")
    
    # Save vectorizer
    Path(vec_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, vec_path)
    print(f"Vectorizer saved to: {vec_path}")
    
    # 5) Train models with timing
    print("Training Naive Bayes...")
    start_time = time.time()
    nb = MultinomialNB()
    nb.fit(X_tr, y_train)
    nb_time = time.time() - start_time
    print(f"Naive Bayes training time: {nb_time:.2f} seconds")
    joblib.dump(nb, model_paths["nb"])
    
    print("\nTraining Logistic Regression...")
    start_time = time.time()
    # Simplified working configuration (matches notebook results)
    lr = LogisticRegression(
        max_iter=500,               # Sufficient for convergence
        random_state=random_state,
        C=1.0                       # Default regularization
    )
    
    lr.fit(X_tr, y_train)
    lr_time = time.time() - start_time
    print(f"Logistic Regression training time: {lr_time:.2f} seconds")
    print(f"Completed iterations: {lr.n_iter_}")
    joblib.dump(lr, model_paths["lr"])
    
    # 6) Evaluate models on test set
    print("\n=== Model Performance ===")
    y_pred_nb = nb.predict(X_te)
    y_pred_lr = lr.predict(X_te)
    
    nb_accuracy = accuracy_score(y_test, y_pred_nb)
    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    
    print(f"Naive Bayes Accuracy: {nb_accuracy:.3f}")
    print(f"Logistic Regression Accuracy: {lr_accuracy:.3f}")
    
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
    with open(summary_path, 'w') as f:
        json.dump(performance_summary, f, indent=2)
    
    print(f"\n💾 Performance summary saved to: {summary_path}")
    print(f"🎯 Production ready: {performance_summary['production_ready']}")
    print(f"📊 Recommended model: {performance_summary['recommended_model']}")
    
    return X_te, y_test, performance_summary