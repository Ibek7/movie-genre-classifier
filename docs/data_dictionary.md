# Data Dictionary

## Raw Dataset (`data/raw/wiki_movie_plots_deduped.csv`)

| Column             | Type                      | Description                                                                 |
|--------------------|---------------------------|-----------------------------------------------------------------------------|
| **Release Year**   | integer                   | Year the movie was released                                                 |
| **Title**          | string                    | Official title of the movie                                                 |
| **Origin/Ethnicity** | string                  | Country or film industry (e.g., American, Bollywood, Tamil, etc.)            |
| **Director**       | string                    | Director name(s). If multiple, separated by commas.                         |
| **Cast**           | string                    | Principal cast members. If multiple, separated by commas.                   |
| **Genre**          | string (pipe-separated)   | One or more genres delimited by `\|` (e.g., `Drama\|Romance\|Comedy`).      |
| **Wiki Page**      | string                    | URL of the Wikipedia page from which the plot was scraped                    |
| **Plot**           | string                    | Full textual synopsis of the movie's storyline                               |

---

## Processed Dataset (`data/processed/cleaned_plots.csv`)

Produced by `src.preprocessing.cleaner.clean_and_save`. Contains a subset of raw columns after deduplication, null removal, and text normalisation.

| Column    | Type                      | Description                                                                 |
|-----------|---------------------------|-----------------------------------------------------------------------------|
| **Title** | string                    | Movie title (unchanged from raw).                                           |
| **Plot**  | string                    | Normalised plot text: lowercased, HTML stripped, punctuation removed.       |
| **Genre** | string (pipe-separated)   | Original genre string; primary label extracted at training time.            |

---

## Genre Label Convention

During training the **first** pipe-separated value is used as the class label. Genres with fewer than `--min-genre-samples` occurrences (default 100) are reassigned to the `"other"` bucket.

Common top-level genres in the Wikipedia dataset:

| Genre    | Approx. share |
|----------|---------------|
| Drama    | ~30 %         |
| Comedy   | ~15 %         |
| Action   | ~10 %         |
| Romance  | ~8 %          |
| Thriller | ~6 %          |
| Horror   | ~5 %          |
| other    | remainder     |

---

## Model Artefacts (`models/`)

| File pattern                    | Description                                          |
|---------------------------------|------------------------------------------------------|
| `production_vectorizer.joblib`  | Fitted `TfidfVectorizer` used for inference          |
| `nb.joblib`                     | Trained `MultinomialNB` classifier                   |
| `lr.joblib`                     | Trained `LogisticRegression` classifier              |
| `performance_summary_<ts>.json` | Accuracy, timing, and config for a training run      |
