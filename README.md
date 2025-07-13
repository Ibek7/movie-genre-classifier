# Movie Genre Classifier  
*A scalable NLP pipeline to predict movie genres from plot summaries*

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