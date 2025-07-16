#!/usr/bin/env python3
"""
Simplified test using the exact working configuration from the notebook
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

print("🎯 Simple Working Configuration Test")
print("=" * 40)

# Load and prepare data
df = pd.read_csv("data/processed/cleaned_plots.csv")
genres = df["Genre"].str.split("|").apply(lambda g: g[0])

# Genre consolidation
genre_counts = genres.value_counts()
top_genres = genre_counts.head(15).index
genres_consolidated = genres.where(genres.isin(top_genres), other="other")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["Plot"], genres_consolidated, test_size=0.2, random_state=42, 
    stratify=genres_consolidated
)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X_tr = vectorizer.fit_transform(X_train)
X_te = vectorizer.transform(X_test)

print(f"Features: {X_tr.shape[1]}, Classes: {len(set(y_train))}")

# Train Naive Bayes
nb = MultinomialNB()
nb.fit(X_tr, y_train)
nb_acc = accuracy_score(y_test, nb.predict(X_te))

# Train simple Logistic Regression
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_tr, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_te))

print(f"NB Accuracy: {nb_acc:.3f}")
print(f"LR Accuracy: {lr_acc:.3f}")

# Save working models
joblib.dump(vectorizer, "models/simple_vectorizer.joblib")
joblib.dump(nb, "models/simple_nb.joblib") 
joblib.dump(lr, "models/simple_lr.joblib")

print("✅ Simple models saved and working!")

# Test prediction
test_plot = "A thrilling action movie with explosions"
test_vec = vectorizer.transform([test_plot])
pred = lr.predict(test_vec)[0]
print(f"Test prediction: {pred}")
