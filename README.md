# Movie Genre Classifier  
*A scalable NLP pipeline to predict movie genres from plot summaries*

## 🚀 **Latest: v2.0.0 - Production-Ready Optimization** 

**Major Performance Breakthrough:** 95%+ speed improvement with production-ready accuracy!

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Training Time** | Minutes | ~10 seconds | **95%+ faster** |
| **Model Accuracy** | Variable | 55.5% (LR) | **Production-ready** |
| **Feature Count** | 100k+ | 5,000 optimized | **95% reduction** |
| **Classes** | 2,227 fragmented | 16 meaningful | **99% consolidation** |
| **Cross-validation** | Unstable | ±0.6% | **Highly stable** |

**🎯 Ready for deployment with comprehensive evaluation framework!**

---

## Table of Contents
1. [Motivation](#motivation)  
2. [Getting Started](#getting-started)  
3. [Pipeline Overview](#pipeline-overview)  
4. [Project Structure](#project-structure)  
5. [How to Run Tests](#how-to-run-tests)  

---

## Motivation
Modern streaming platforms and recommendation systems hinge on accurately tagging content by genre. While metadata often exists, it can be incomplete or inconsistent—especially for indie films or user-generated uploads.  
Our **Movie Genre Classifier** leverages advances in natural language processing and classic machine-learning models to:  
- **Automatically infer** one or more genres from a film’s plot synopsis  
- **Reduce manual labeling effort** and correct metadata errors  
- **Enhance discoverability** for niche and emerging titles  

By transforming raw plot text into actionable insights, we empower content providers, archivists, and researchers to categorize large libraries quickly, consistently, and at scale.

---

## Getting Started

### 1. Prerequisites
- **Python 3.9+**  
- A modern UNIX-like shell (macOS/Linux) or Git Bash on Windows  
- (Optional) [Docker](https://www.docker.com/) for containerized setup  

### 2. Clone & Setup
```bash
# From your workspace directory:
git clone https://github.com/your-username/movie-genre-classifier.git
cd movie-genre-classifier

# Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Optional: install developer tooling
pip install -r requirements-dev.txt
```

## Quick Start

### Production Pipeline
```bash
# Run optimized training pipeline
python -m src.main \
    --data-path data/processed/cleaned_plots.csv \
    --vectorizer-path models/production_vectorizer.joblib \
    --nb-model-path models/nb.joblib \
    --lr-model-path models/lr.joblib
```

### Quick Prediction
```python
import joblib

# Load optimized models
vectorizer = joblib.load('models/simple_vectorizer.joblib')
model = joblib.load('models/simple_lr.joblib')

# Predict genre
plot = "A thrilling action movie with explosions and car chases"
genre = model.predict(vectorizer.transform([plot]))[0]
print(f"Predicted genre: {genre}")
```

### CLI Prediction
```bash
# Predict for one or more raw plot strings
python -m src.models.predict text \
    --plots "A detective investigates a murder" "A family survives a haunted house" \
    --vectorizer models/production_vectorizer.joblib \
    --model models/lr.joblib

# Predict for all rows in a CSV with a Plot column
python -m src.models.predict csv \
    --input-csv data/processed/cleaned_plots.csv \
    --output-csv data/processed/predictions.csv \
    --vectorizer models/production_vectorizer.joblib \
    --model models/lr.joblib
```

### Installed CLI Commands
After installing the package (e.g., `pip install -e .`), you can use:

```bash
mgc-train --data-path data/processed/cleaned_plots.csv
mgc-predict text --plots "A detective investigates a murder" \
    --vectorizer models/production_vectorizer.joblib \
    --model models/lr.joblib
```

Notes:
- `--test-size` must be between `0` and `1` (exclusive).
- `--max-features` and `--min-genre-samples` must be positive integers.
- Prediction input must include at least one non-empty plot summary.

## How to Run Tests

```bash
# Run all tests
pytest -q

# Run focused test modules
pytest -q tests/test_models.py tests/test_vectorization.py tests/test_predict.py
```

## Makefile Shortcuts

```bash
make install    # install runtime deps + editable package
make test       # run full test suite
make test-fast  # run focused model/prediction tests
make lint       # compile-check Python modules
make clean      # remove Python cache files
```

## Developer Helper Functions

Recent utility additions for notebook and script ergonomics:

- `src.preprocessing.cleaner.average_word_length(text)` — returns mean token length.
- `src.preprocessing.tokenizer.unique_tokens(tokens)` — drops duplicate tokens while preserving order.
- `src.utils.helpers.safe_divide(numerator, denominator, default=0.0)` — avoids zero-division boilerplate.
- `src.utils.helpers.clamp_probability(value)` — constrains scores to the `[0, 1]` interval.
- `src.features.vectorizer.describe_vectorizer(vectorizer)` — summarizes fitted state and TF-IDF config.

---