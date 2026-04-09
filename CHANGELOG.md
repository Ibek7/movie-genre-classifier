# Changelog

All notable changes to **movie-genre-classifier** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `src/constants.py` — centralised genre labels, default filenames, and
  hyper-parameter defaults (`TOP_GENRES`, `MIN_GENRE_SAMPLES`, `DEFAULT_MAX_FEATURES`, etc.).
- `predict_with_confidence()` in `src/models/predict.py` — returns the top
  predicted genre together with its probability score for each plot.
- `plot_confusion_matrix()` in `src/utils/visualization.py` — annotated
  heatmap for quick inspection of per-class error patterns.
- `plot_precision_recall_per_class()` in `src/utils/visualization.py` — grouped
  bar chart comparing precision and recall across genre classes.
- `compute_classification_report()` in `src/utils/helpers.py` — thin wrapper
  around `sklearn.metrics.classification_report` returning a JSON-serialisable dict.
- `word_count()` and `sentence_count()` in `src/preprocessing/cleaner.py` — lightweight
  text-statistics helpers useful for EDA filtering and quality checks.
- Unit tests for `word_count` and `sentence_count` in `tests/test_preprocessing.py`.
- Sections 9–11 in `docs/methodology.md` covering text statistics, observability
  (structured logging), and the new constants module.

### Changed
- `src/models/train.py` — replaced all `print()` calls with structured
  `logging.getLogger(__name__)` calls (`INFO` for progress, `DEBUG` for diagnostics).
- `src/utils/visualization.py` — added `numpy` and `sklearn.metrics` imports
  required by the new chart helpers.

---

## [0.2.0] — 2025-07-16

### Added
- `batch_predict_from_dir()` for bulk CSV directory inference.
- `get_top_genres()` utility for inspecting genre frequency distributions.
- `GridSearchCV` hyperparameter-tuning tests for LR and NB pipelines.
- Root `__init__.py` populated with project overview and quick-start examples.
- Extended `.gitignore` with build dirs, packaging artefacts, IDE files, and media patterns.
- Extended `docs/data_dictionary.md` with processed dataset, genre convention, and artefacts sections.
- `tests/conftest.py` with shared session-scoped pytest fixtures.
- End-to-end integration tests covering clean → train → predict pipeline.
- `save_vectorizer` / `load_vectorizer` helpers in `src/features/vectorizer.py`.

### Changed
- `train_and_save_models` docstring replaced inline comments with proper NumPy-style docstring.

---

## [0.1.0] — 2025-07-15

### Added
- Initial project skeleton: `src/`, `tests/`, `notebooks/`, `data/`, `docs/`, `models/`.
- Preprocessing pipeline (`cleaner.py`, `tokenizer.py`).
- TF-IDF feature engineering (`vectorizer.py`).
- Baseline Naive Bayes and Logistic Regression training (`train.py`).
- Inference helpers (`predict.py`).
- Evaluation notebook and model-comparison utilities.
- `Dockerfile` for containerised inference.
- `setup.py` / `requirements.txt` for packaging.
