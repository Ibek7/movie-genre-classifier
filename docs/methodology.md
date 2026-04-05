# Methodology

This document describes the end-to-end approach used to build the movie-genre classifier.

---

## 1. Data Sources

| Dataset | Description |
|---------|-------------|
| `wiki_movie_plots_deduped.csv` | Wikipedia movie plots with genre labels |
| CMU Movie Summary Corpus | Alternative plot summaries and metadata |

The Wikipedia dataset was chosen as the primary source due to its size and genre coverage.

---

## 2. Data Cleaning (`src/preprocessing/cleaner.py`)

Raw data passes through four deterministic steps:

1. **Load** — read CSV with `pandas`.
2. **Deduplicate** — drop exact duplicates on `(Title, Plot)`.
3. **Prune missing** — remove rows where `Plot` or `Genre` is null.
4. **Normalise text** — lowercase, strip HTML tags, remove punctuation, collapse whitespace.

The cleaned file is written to `data/processed/cleaned_plots.csv`.

---

## 3. Genre Consolidation (`src/models/train.py`)

Raw genre labels are multi-valued (`Action|Drama`).  The primary label (first pipe-separated value) is used.  Rare genres (fewer than `--min-genre-samples` occurrences, default 100) are merged into an `"other"` bucket.  This reduces the class count from ~50+ to a manageable 15–20, improving both model accuracy and class balance.

---

## 4. Feature Engineering (`src/features/vectorizer.py`)

TF-IDF with tuned defaults:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_features` | 5 000 | Balances vocabulary size vs. speed |
| `ngram_range` | (1, 1) | Unigrams shown sufficient in experiments |
| `stop_words` | English | Remove uninformative high-frequency words |
| `max_df` | 0.95 | Ignore terms appearing in >95 % of documents |
| `min_df` | 15 | Ignore very rare terms |

---

## 5. Model Training (`src/models/train.py`)

Two classifiers are trained and compared:

- **Multinomial Naive Bayes** — fast baseline; works well with TF-IDF counts.
- **Logistic Regression** — stronger discriminative model; `C=1.0`, `max_iter=500`.

A stratified 80/20 train/test split is used (`--test-size 0.2`, `--random-state 42`).

---

## 6. Hyperparameter Tuning (`src/models/tune_hyperparameters.py`)

`GridSearchCV` pipelines explore:

- TF-IDF `max_features` ∈ {3 000, 5 000, 10 000}
- `ngram_range` ∈ {(1,1), (1,2)}
- LR `C` ∈ {0.1, 1.0, 10.0} / NB `alpha` ∈ {0.01, 0.1, 1.0}

3-fold CV is used to select the best configuration.

---

## 7. Evaluation

Models are evaluated on the held-out test set using **accuracy**.  A performance JSON summary is saved to `models/performance_summary_<timestamp>.json` after every training run.  The recommended model is whichever achieves the higher test accuracy.

---

## 8. Inference (`src/models/predict.py`)

Three public helpers are provided:

| Function | Use-case |
|----------|----------|
| `predict(plots, ...)` | Batch list of raw plot strings → genre labels |
| `predict_genre(plot, ...)` | Single plot → genre label |
| `predict_proba(plots, ...)` | Batch → per-class probability dicts |
| `predict_from_csv(...)` | CSV in, CSV out (batch inference) |
