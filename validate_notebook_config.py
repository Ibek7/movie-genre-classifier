#!/usr/bin/env python3
"""
Quick validation test using exact notebook configuration
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

print("🧪 Quick Validation Test - Exact Notebook Config")
print("=" * 50)

# Load data exactly like notebook
df = pd.read_csv("data/processed/cleaned_plots.csv")
genres = df["Genre"].str.split("|").apply(lambda g: g[0])

# Apply same genre consolidation as notebook
min_samples = 100
genre_counts = genres.value_counts()
common_genres_100 = genre_counts[genre_counts >= min_samples].index
top_genres = genre_counts.head(15).index

if len(common_genres_100) <= 15:
    chosen_genres = common_genres_100
    method = f"genres with {min_samples}+ samples"
else:
    chosen_genres = top_genres
    method = "top 15 genres"

genres_consolidated = genres.where(genres.isin(chosen_genres), other="other")

print(f"Using {method}: {len(genres_consolidated.value_counts())} classes")

# Train/test split exactly like notebook
X = df["Plot"]
y = genres_consolidated
stratify_arg = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_arg
)

# TF-IDF exactly like notebook
max_features = 5000
vectorizer = TfidfVectorizer(max_features=max_features)
X_tr = vectorizer.fit_transform(X_train)
X_te = vectorizer.transform(X_test)

print(f"TF-IDF features: {X_tr.shape[1]}")
print(f"Training samples: {X_tr.shape[0]}")
print(f"Unique classes: {len(set(y_train))}")

# Train models exactly like notebook
print("Training Naive Bayes...")
start_time = time.time()
nb = MultinomialNB()
nb.fit(X_tr, y_train)
nb_time = time.time() - start_time
print(f"Naive Bayes training time: {nb_time:.2f} seconds")

print("\nTraining Logistic Regression...")
start_time = time.time()
lr = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1, class_weight="balanced", random_state=42)
lr.fit(X_tr, y_train)
lr_time = time.time() - start_time
print(f"Logistic Regression training time: {lr_time:.2f} seconds")

# Get predictions and accuracies
y_pred_nb = nb.predict(X_te)
y_pred_lr = lr.predict(X_te)

nb_accuracy = accuracy_score(y_test, y_pred_nb)
lr_accuracy = accuracy_score(y_test, y_pred_lr)

print(f"\n=== Results ===")
print(f"Naive Bayes Accuracy: {nb_accuracy:.3f}")
print(f"Logistic Regression Accuracy: {lr_accuracy:.3f}")

# Save the working models
joblib.dump(vectorizer, "models/validated_vectorizer.joblib")
joblib.dump(nb, "models/validated_nb_model.joblib")
joblib.dump(lr, "models/validated_lr_model.joblib")

print(f"\n✅ Validated models saved!")
print(f"🎯 LR Production Ready: {lr_accuracy > 0.4}")
