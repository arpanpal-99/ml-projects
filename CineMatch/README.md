# CineMatch

CineMatch helps you discover movies that share descriptive details with a movie you
already like. Search for a title, choose the correct year/ID, and browse up to five
close matches. After you build the local catalog and model, movie search and
recommendation ranking work without an internet connection or a TMDB API key.
Loading poster images does require an internet connection.

> **Important:** This project does not include the movie datasets or generated model
> artifacts. Add source CSV files that you are entitled to use before building the
> app. It does not invent sample catalog entries when the datasets are missing.

## What you need

- Python 3.10 or newer
- The TMDB 5000 Movies and Credits CSV files
- Optional: a TMDB v3 API key. It is used to look up a poster when a movie has no
  local poster path. Runtime is shown when available in the local movie data.

## Run it on Windows

Open PowerShell in this project folder and run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

`requirements-dev.txt` includes `requirements.txt`, so this installs both the
application libraries and the test/lint tools. If you only want to run the app,
install just its runtime libraries instead:

```powershell
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass`
in that PowerShell window, then activate the environment again. You can also use
`.venv\Scripts\python.exe` and `.venv\Scripts\streamlit.exe` directly without
activating it.

## Get the data and build the movie catalog

Obtain the **TMDB 5000 Movie Dataset** distribution from its publisher, for example
the [TMDB Movie Metadata dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
Check the source page for its current license, attribution requirements, and
redistribution terms. Do not publish the source CSVs or generated catalog unless
those terms allow it.

Place the files in `data/raw/` using these exact names:

```text
data/raw/tmdb_5000_movies.csv
data/raw/tmdb_5000_credits.csv
```

Then build the local data and model artifacts:

```powershell
python -m scripts.build_artifacts
```

The command validates required columns and unique numeric IDs, joins the datasets
by movie ID, extracts genres/keywords/cast/directors, builds movie tags, and creates
a sparse text-feature matrix. It writes these files into `artifacts/`:

- `movies.parquet` — clean movie records and display metadata
- `vectorizer.joblib` — fitted CountVectorizer
- `feature_matrix.npz` — compressed sparse movie-by-word matrix
- `manifest.json` — schema, ordering checksum, dimensions, build summary, and file
  integrity checks

The builder reports missing input files, invalid or duplicate IDs, and missing
required columns. It records unmatched movie/credit IDs and rows excluded because
they have no usable title in the manifest's build summary. Malformed JSON-like
metadata is treated as empty and counted; if its rate exceeds the configured limit,
the build stops. Resolve a data-format problem or deliberately adjust
`CINEMATCH_MAX_PARSE_FAILURE_RATE` in your environment. The default malformed-
metadata limit is 5%.

## Start CineMatch

```powershell
streamlit run app.py
```

Open the local address Streamlit prints (usually `http://localhost:8501`). Start in
the **Search the catalogue** picker: type a title, choose the right release, and
select it to open that exact movie's profile in a modal. Recommendations stay
hidden until you click **Find similar movies**. Use the picker’s built-in clear
control to dismiss the preview and hide the recommendation section.
**Surprise me** starts a trail from a quality-aware popular title and opens its
details modal. Choose
**Details** on any movie card to open its profile in a centered modal over the
blurred page; close it to return to the same place in the catalogue.
Selection, details, and genre changes use distinct contextual loading messages.
Dismissed movie profiles do not reopen during later page interactions.
After starting a recommendation with **Find similar movies** or **Surprise me**,
the **Continue exploring** row offers compact buttons for recently selected movies
and stays visible for the rest of the session, even when the picker is cleared.

The home screen has two deterministic discovery rails rather than random featured
cards:

- **Top picks for tonight** rotates a random selection of titles rated 7.5 or higher
  with available posters from two or three genres. Choosing a specific genre narrows
  this shelf to that genre.
- **Popular across the catalogue** rotates titles rated 8.3 or higher across all
  genres, regardless of the genre filter; titles without available posters are skipped.

On desktop, each shelf displays up to eight titles in two rows of four compact
cards. Card titles stay on one line and scroll horizontally when they do not fit;
movie descriptions are kept out of the cards and shown in the Details modal.
Use the genre filter to narrow Top picks and **Show more picks** to refresh both
random selections. Each card can become the next recommendation, open a movie profile, be
saved to **My List**, or be added to the session-only taste profile. Liking more
than one title creates a blended **More like what you like** rail; no account or
personal data service is required. The **Closest match**, **Balanced**, and
**Hidden gems** modes make the ranking trade-off explicit, and cards explain shared
genres, cast, directors, or themes where that metadata exists. If a local poster
path is missing, CineMatch looks it up through TMDB when a key is configured;
otherwise it displays a designed title cover.

Artifacts are loaded once and cached for the running app. Rebuild them only when you
change the input data or model configuration. The app checks artifact versions,
file hashes, movie ordering, and matrix dimensions at startup; if something is
missing or inconsistent, rebuild with `python -m scripts.build_artifacts`.

## TMDB posters and credentials

The TMDB 5000 Movies CSV includes a `poster_path` field, but some movie records may
not have a local poster path. When a TMDB key is configured, CineMatch looks up and
caches poster paths through the TMDB movie details endpoint; displaying the
resulting image still requires an internet connection. Without a key or when a
lookup fails, CineMatch uses a designed title cover so discovery cards remain
complete rather than showing a broken image. To enable official poster lookups,
copy `.env.example` to `.env` and set your TMDB **v3 API key**:

```dotenv
TMDB_API_KEY=your_key_here
```

Never commit `.env` or paste a real key into source code. For a hosted Streamlit
app, store the key in the host's secret manager, for example
`.streamlit/secrets.toml` during local testing:

```toml
TMDB_API_KEY = "your_key_here"
```

The client uses timeouts, bounded retries, a small TTL cache, and graceful failure.
It never logs credentials. The app remains usable when the API key is absent or the
service is unavailable.

## Run tests and quality checks

The tests use small in-memory movie fixtures and mocked HTTP calls; they do not
download datasets or contact TMDB:

```powershell
pytest
ruff check .
```

To check that the source-data error is clear before downloading the datasets:

```powershell
python -m scripts.build_artifacts
```

Run this check only while the CSVs are absent; it should exit with an actionable
missing-file message. If both files are present, the command builds or replaces the
artifacts as described above.

## How recommendations work

For each movie, CineMatch combines genres, keywords, the principal cast, directors,
and the overview into a normalized text field called `tags`. A fitted
`CountVectorizer` turns those tags into a sparse bag-of-words vector. The core
matching signal is cosine similarity between a selected movie (or the average of a
small set of liked movies) and other catalogue vectors.

The **Closest match** mode preserves that pure similarity order. **Balanced** keeps
similarity dominant while adding modest, normalized quality and popularity signals;
**Hidden gems** puts more weight on quality and rewards less broadly popular titles.
Top-pick shelves use a separate Bayesian rating so a single 10/10 vote does not
outrank a well-supported favourite. Every sort has stable title/ID tie-breaks.

This is still **content similarity**, not a prediction that you will enjoy a movie.
It does not learn from user ratings, watch history, or audience behavior. Likes and
saves stay in the current browser session, and films with no useful text features do
not receive arbitrary recommendations.

## Project layout

```text
app.py                         Streamlit interface
src/data/                      CSV validation, metadata parsing, and tags
src/features/                  CountVectorizer feature construction
src/recommender/                Ranking and artifact validation/persistence
src/services/                  Optional TMDB client
ml_pipeline.py                 Commented ML build/load/recommendation workflow
app.py                         Streamlit UI and session interactions
scripts/build_artifacts.py     CLI entry point for the offline model build
tests/                          Offline unit and artifact tests
data/raw/                       User-supplied source CSVs (not committed)
artifacts/                      Generated model files (not committed)
```

The ML workflow is deliberately separate from the Streamlit interface. Start with
`ml_pipeline.py` to follow how source data becomes normalized movie features and
how those features are used to rank recommendations. The detailed implementations
remain in `src/data/`, `src/features/`, and `src/recommender/`; `app.py` calls the
pipeline and focuses on the visual experience and browser-session features.

## Deployment

CineMatch is a regular Streamlit application and can be deployed on a
Streamlit-compatible host. For a hosted release:

1. Use the same supported Python version and pinned packages from
   `requirements.txt`.
2. Build artifacts from the approved source data before deployment. Keep generated
   model files in a private/trusted location; `joblib` files can execute code when
   loaded and must never be accepted from untrusted users.
3. Ensure the four files in `artifacts/` are available to the app at startup. The
   source CSVs are only needed while building the artifacts, not to serve
   recommendations.
4. Configure optional secrets in the hosting provider's secret manager, not in
   source control.
5. Check the dataset's and TMDB's current terms and attribution requirements before
   distributing data or operating a public instance.
6. Smoke-test startup and recommendations both with and without the TMDB key.

Paths can be customized with `CINEMATCH_ROOT`, `CINEMATCH_RAW_DATA_DIR`, and
`CINEMATCH_ARTIFACTS_DIR`. Model settings can be adjusted with
`CINEMATCH_MAX_FEATURES`, `CINEMATCH_MAX_PARSE_FAILURE_RATE`, and
`CINEMATCH_PRINCIPAL_CAST_SIZE`.

## Attribution

Movie metadata originates from the TMDB 5000 dataset distribution; posters and
optional metadata are associated with TMDB. Follow the terms provided by the
dataset publisher and the [TMDB developer terms and documentation](https://developer.themoviedb.org/docs).
This product uses the TMDB API but is not endorsed or certified by TMDB.
