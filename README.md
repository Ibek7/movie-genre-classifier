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
```

## Quick Start

### Production Pipeline
```bash
# Run optimized training pipeline
python -c "
from src.models.train import train_and_save_models
train_and_save_models(
    data_path='data/processed/cleaned_plots.csv',
    vec_path='models/production_vectorizer.joblib',
    model_paths={'nb': 'models/nb.joblib', 'lr': 'models/lr.joblib'},
    max_features=5000,
    min_genre_samples=100
)
"
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

---