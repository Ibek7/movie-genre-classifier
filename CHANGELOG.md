# Changelog

All notable changes to **movie-genre-classifier** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `detokenize()` in `src/preprocessing/tokenizer.py` for rebuilding text from token lists.
- `safe_divide()` in `src/utils/helpers.py` for zero-safe ratio calculations.
- `describe_vectorizer()` in `src/features/vectorizer.py` for compact TF-IDF metadata inspection.
- `average_word_length()` in `src/preprocessing/cleaner.py` for quick lexical-complexity checks.
- `unique_tokens()` in `src/preprocessing/tokenizer.py` for order-preserving token deduplication.
- `clamp_probability()` in `src/utils/helpers.py` to constrain scores to the `[0, 1]` range.
- Unit tests for `detokenize()` in `tests/test_preprocessing.py`.
- Unit tests for `safe_divide()` in `tests/test_utils.py`.
- Unit tests for `describe_vectorizer()` in `tests/test_vectorization.py`.
- Unit tests for `average_word_length()` and `unique_tokens()` in `tests/test_preprocessing.py`.
- Unit tests for `clamp_probability()` in `tests/test_utils.py`.

### Changed
- `src/preprocessing/__init__.py` now exports tokenizer utilities and text-stat helpers.
- `src/models/predict.py` now centralizes artifact path checks via `validate_artifact_paths()`.
- `docs/methodology.md` now includes a dedicated section for developer ergonomics helpers.
- `src/preprocessing/__init__.py` now also exports `average_word_length` and `unique_tokens`.
- `src/utils/__init__.py` now exports `safe_divide` and `clamp_probability`.
- `README.md` now includes a dedicated "Developer Helper Functions" section.

---

## [0.3.0] — 2025-07-16

### Added
- `filter_short_plots()` in `src/preprocessing/cleaner.py` — removes DataFrame rows
  whose plot word-count falls below a configurable minimum (default 20 words).
- `predict_top_k()` in `src/models/predict.py` — returns the top-*k* genre candidates
  with per-class probability scores for each plot, sorted by descending confidence.
- `format_duration()` and `elapsed_time()` in `src/utils/helpers.py` — human-readable
  time formatting helpers for surfacing pipeline timing information.
- `plot_learning_curve()` in `src/utils/visualization.py` — plots training vs.
  validation accuracy across training-set sizes.
- `plot_f1_heatmap()` in `src/utils/visualization.py` — renders per-class F1 scores
  from a `classification_report` dict as an annotated colour heatmap.
- `src/utils/__init__.py` — public API surface for the utils sub-package, exporting
  all 8 helpers and 6 visualisation functions via `__all__`.
- `src/constants.py` — centralised genre labels, default filenames, and
  hyper-parameter defaults (`TOP_GENRES`, `MIN_GENRE_SAMPLES`, `DEFAULT_MAX_FEATURES`, etc.).
- `predict_with_confidence()` in `src/models/predict.py` — returns the top predicted
  genre together with its probability score for each plot.
- `plot_confusion_matrix()` in `src/utils/visualization.py` — annotated heatmap for
  quick inspection of per-class error patterns.
- `plot_precision_recall_per_class()` in `src/utils/visualization.py` — grouped bar
  chart comparing precision and recall across genre classes.
- `compute_classification_report()` in `src/utils/helpers.py` — thin wrapper around
  `sklearn.metrics.classification_report` returning a JSON-serialisable dict.
- `word_count()` and `sentence_count()` in `src/preprocessing/cleaner.py` — lightweight
  text-statistics helpers useful for EDA filtering and quality checks.
- `truncate_plot()` in `src/preprocessing/cleaner.py` — clips plots to a maximum
  word count to cap vectorizer input length.
- Sections 9–11 in `docs/methodology.md` covering text statistics, observability
  (structured logging), and the new constants module.
- Unit tests for `word_count`, `sentence_count`, `truncate_plot`, and
  `filter_short_plots` in `tests/test_preprocessing.py`.
- Unit tests for `predict_with_confidence` and `predict_top_k` in
  `tests/test_predict.py`, including a shared `tiny_model` fixture.
- 14 unit tests in `tests/test_constants.py` covering all exported constants.
- Save/load round-trip and vocabulary-size tests in `tests/test_vectorization.py`.
- Classification-report and top-genres tests in `tests/test_utils.py`.

### Changed
- `src/models/__init__.py` — added `predict_with_confidence` and `predict_top_k`
  to its public exports; expanded module docstring.
- `src/models/train.py` — replaced all `print()` calls with structured
  `logging.getLogger(__name__)` calls (`INFO` for progress, `DEBUG` for diagnostics).
- `src/utils/visualization.py` — added `numpy`, `sklearn.metrics`, and `Dict` imports
  required by the new chart helpers.
- `src/__init__.py` — bumped `__version__` to `"0.3.0"`; added `__email__`,
  `__description__`, `__license__`, `__url__`, and `__all__` listing all sub-packages.

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
