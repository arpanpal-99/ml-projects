# Movie Recommendation System

A small, local-first teaching project that recommends movies using basic
content-based machine learning. The Streamlit interface is separate from the
data-processing and recommendation pipeline, and no generated model artifacts
are saved.

## Run on Windows

From this folder in PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

Using `python -m streamlit` starts Streamlit through the selected Python
environment, avoiding stale Windows console launchers.

## TMDB API key

Movie details and posters use the TMDB API. Add your key to this project's
`.env` file:

```text
TMDB_API_KEY=your_tmdb_api_key_here
```

The local `.env` is ignored by Git. `.env.example` shows the required variable
without containing a real credential.

## Dataset

Place these files in the project's `data/raw/` directory:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Download the TMDB 5000 Movie Dataset from its publisher. Check the source page
for current licensing, attribution, and redistribution terms; the CSVs are
intentionally excluded from Git.

## How it works

1. **Load and join:** pandas reads the movie and credits files. Credits are joined
   to movie metadata using movie ID.
2. **Feature engineering:** JSON-like genres and keywords, the first three cast
   members, directors, and the overview are combined into a lowercase `tags`
   string for each movie. Names are joined with underscores so multi-word names
   can be treated as one feature token.
3. **Vectorize:** scikit-learn's `CountVectorizer` learns up to 5,000 word
   features, ignores common English stop words, and converts each movie into a
   sparse bag-of-words vector.
4. **Compare:** when a movie is selected, cosine similarity compares its vector
   with all catalogue vectors. The selected movie is excluded and the highest
   positive matches are shown.

This is an **unsupervised, content-based recommender**, not a model trained on
people's ratings. Shared words are a simple proxy for related content; similarity
is not a guarantee that a viewer will like a recommendation. The app computes one
movie's scores at a time instead of storing a full all-pairs similarity matrix.

## Project files

- `app.py` - Streamlit interface, styling, and movie detail display.
- `ml_pipeline.py` - dataset loading, feature engineering, and recommendation
  ranking.
- `.env.example` - the TMDB environment variable template; put your key in the
  ignored `.env` file.
- `requirements.txt` - the direct runtime dependencies.
- `data/raw/.gitkeep` - keeps the local dataset directory visible without
  committing the dataset.
