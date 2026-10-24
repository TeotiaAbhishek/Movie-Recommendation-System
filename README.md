# NLP-Powered Movie Recommendation System

Describe the kind of film you want in plain English, or name one you liked, and get recommendations that match the meaning of your request, not just its keywords. The system searches 44,471 films using sentence embeddings and approximate nearest neighbour search, then re-ranks results by audience sentiment, popularity and variety. It runs as a Streamlit app and a FastAPI service, with tests, CI and a Docker image.

> "Something dark and psychological with a mystery theme"
>
> "Films like John Wick"

Built as a team project for 36118 Applied Natural Language Processing at the University of Technology Sydney.

## How it works

```text
Movie metadata + user reviews
        |
Review sentiment (VADER), recency-weighted per movie
        |
Text per movie: title, overview, genres, review snippets
        |
SBERT embeddings (all-MiniLM-L6-v2, 384 dimensions)
        |
HNSW cosine index (hnswlib)
        |
Query: free text or a movie title
        |
Top 200 nearest neighbours
        |
Score fusion: 0.7 similarity + 0.2 sentiment + 0.1 log-popularity
        |
MMR re-ranking for variety, top 12 shown
```

**Offline (notebook):** `notebooks/recommender.ipynb` scores each review with VADER, aggregates sentiment per movie with an 18-month half-life so recent reviews count more, shrinks movies with few reviews towards neutral, builds a text description per movie, embeds it and saves an HNSW index. Everything the app needs is saved to `artifacts/`.

**Online (app):** `App/app.py` embeds the query, retrieves 200 candidates from the index, blends similarity with sentiment and popularity, and uses Maximal Marginal Relevance (MMR, lambda 0.7) so the final 12 aren't near-duplicates of each other.

## Features

| Tab | What it does |
|---|---|
| Ask | Type a request in natural language. If it names films, they're found with fuzzy matching (RapidFuzz) or an optional local LLM, and used as seeds; otherwise the text itself is the query. |
| Browse | Filter the catalogue by genre and year range, sorted by popularity, year or title. |
| Similar by title | Pick a film and get the 12 most similar, with posters from TMDb. |

When a local LLM is configured, it also writes a one-line summary of each result set. Without one, the app falls back to fuzzy matching and a rule-based summary, so it works fully offline.

## Data

| Item | Value |
|---|---|
| Movies | 44,471 |
| Sources | TMDb and IMDb metadata, Kaggle movie and review datasets |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Embedding size | 384 |
| Index | HNSW, cosine similarity |

## Results

There are no user ratings in the data, so `python -m movie_recommendation_system.evaluate` measures proxy metrics for similar-title recommendations. It uses 500 random well-known seed movies (100+ votes), 12 recommendations each:

| Ranking | Genre precision@12 | Diversity | Catalogue coverage | Latency p50 / p95 |
|---|---:|---:|---:|---:|
| Similarity only | 0.818 | 0.528 | 10.5% | 3.6 / 5.4 ms |
| + sentiment and popularity | 0.822 | 0.528 | 10.2% | 3.4 / 4.9 ms |
| + MMR (default) | 0.817 | 0.543 | 10.4% | 4.0 / 5.9 ms |

- **Genre precision@12:** share of recommendations that share at least one genre with the seed movie.
- **Diversity:** 1 minus the average cosine similarity between movies in the same list.
- **Latency:** retrieval and ranking time. Free-text search adds the SBERT encoding of the query.

**What this shows:** semantic retrieval does most of the work. About 82% of recommendations share a genre with the seed, and HNSW answers in a few milliseconds. Sentiment and popularity fusion barely changes the lists. MMR lifts diversity by about 3% at almost no cost to relevance. Measuring real quality would need user ratings or click data, which is the next step.

## How to run

Requires Python 3.10+.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build, much smaller
pip install -e ".[app,dev]"

make app        # Streamlit app at localhost:8501
make api        # FastAPI at localhost:8000, docs at /docs
make test       # unit and API tests on a tiny in-memory catalogue (no downloads)
make evaluate   # offline metrics above
```

The prebuilt index, embeddings and metadata in `artifacts/` load at start-up. The sentence model downloads on the first free-text search. Title search and browsing work without it. To rebuild the artifacts, run `notebooks/recommender.ipynb`.

**Optional local LLM:** `pip install -e ".[llm]"` and set `LLM_MODEL_PATH` to a GGUF model file. The app then uses it to extract titles from requests and to summarise results. Without it, the app falls back to quoted phrases, spaCy entities and a rule-based summary.

### API

| Endpoint | Purpose |
|---|---|
| `GET /search?q=slow-burn mystery in a small town&k=12` | free-text recommendations |
| `GET /similar?title=john wick&k=12` | movies similar to a title (typo-tolerant; 404 if nothing matches) |
| `GET /genres` | genre list |
| `GET /health` | status and catalogue size |

### Docker

```bash
docker build -t movie-recommender .
docker run -p 8000:8000 movie-recommender
```

The image bundles the CPU build of PyTorch, the sentence model and the artifacts, so it starts without network access.

### Live demo (Streamlit Community Cloud)

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. Choose **Create app**, pick this repository, branch `main` and main file `App/app.py`, then set Python to 3.11 under Advanced settings.
3. Deploy. `requirements.txt` installs the CPU build of PyTorch to stay within the free tier's memory.

## Testing and CI

GitHub Actions runs on every push and pull request. It checks linting with ruff, runs the tests (ranking maths, MMR, title matching, browse filters and every API endpoint, on a fake 8-movie catalogue) and the offline evaluation on the real catalogue. It then builds the Docker image and smoke-tests `/similar` and `/search`.

## Project structure

```text
├── movie_recommendation_system/
│   ├── recommender.py          loads artifacts; search, similar, browse
│   ├── ranking.py              score fusion and MMR
│   ├── assistant.py            optional LLM / spaCy title extraction and summaries
│   ├── api.py                  FastAPI service
│   └── evaluate.py             offline metrics
├── App/app.py                  Streamlit UI (no ranking logic of its own)
├── artifacts/                  embeddings, HNSW index, sentiment, metadata
├── notebooks/                  data cleaning, sentiment, embeddings and index build
├── tests/
├── Dockerfile
└── .github/workflows/ci.yml
```

## Next steps

- Tune the 0.7 / 0.2 / 0.1 weights against user ratings (for example MovieLens) instead of genre overlap.
- Add collaborative filtering for a hybrid recommender once user histories are available.
- Replace VADER with a transformer sentiment model.

## Team

- **Abhishek Teotia:** recommendation engine, retrieval pipeline, integration and testing
- **Muskan Choudhary:** NLP query extraction
- **Justin Govanxa:** sentiment module
- **Yuzhong Wang:** front end and LLM integration

For educational and research use.
