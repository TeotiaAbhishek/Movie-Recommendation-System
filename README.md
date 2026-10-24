# NLP-Powered Movie Recommendation System

Describe the kind of film you want in plain English, or name one you liked, and get recommendations that match the meaning of your request, not just its keywords. The system searches 44,471 films using sentence embeddings and approximate nearest neighbour search, then re-ranks results by audience sentiment, popularity and variety. It runs as a Streamlit app.

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
Score fusion: 0.7 similarity + 0.2 sentiment + 0.1 popularity
        |
MMR re-ranking for variety, top 12 shown
```

**Offline (notebook):** `notebooks/recommender.ipynb` scores each review with VADER, aggregates sentiment per movie with an 18-month half-life so recent reviews count more, shrinks movies with few reviews towards neutral, builds a text description per movie, embeds it and saves an HNSW index. Everything the app needs is saved to `artifacts/`.

**Online (app):** `App/app.py` embeds the query, retrieves 200 candidates from the index, blends similarity with sentiment and popularity, and uses Maximal Marginal Relevance (MMR, lambda 0.7) so the final 12 aren't near-duplicates of each other.

## Features

| Tab | What it does |
|---|---|
| Home | Type a request in natural language. If it names films, they're found with fuzzy matching (RapidFuzz) or an optional local LLM, and used as seeds; otherwise the text itself is the query. |
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

## How to run

Requires Python 3.11.

```bash
pip install streamlit pandas numpy scikit-learn sentence-transformers hnswlib rapidfuzz requests pyarrow pillow spacy
python -m spacy download en_core_web_sm   # optional, improves title extraction

cd App
streamlit run app.py
```

The prebuilt index and embeddings in `artifacts/` are loaded at start-up, so there is no training step. To rebuild them, run `notebooks/recommender.ipynb`.

**Optional local LLM:** install `llama-cpp-python` and point `LLM_MODEL_PATH` to a GGUF model file to enable LLM title extraction and summaries.

```bash
export LLM_MODEL_PATH=/path/to/model.gguf
```

## Project structure

```text
├── App/
│   ├── app.py                  Streamlit app: search, ranking and UI
│   └── static/no_poster.png
├── artifacts/
│   ├── embeddings.npy          384-d embedding per movie
│   ├── hnsw_cosine.idx         HNSW index
│   ├── review_sentiment_01.npy sentiment per movie, scaled 0 to 1
│   ├── movies_aug.parquet      movie metadata
│   ├── id_map.json
│   └── config.json
├── notebooks/
│   ├── movie.ipynb             data exploration and cleaning
│   └── recommender.ipynb       sentiment, embeddings and index build
└── pyproject.toml
```

## Next steps

- Measure ranking quality offline (precision@k against held-out user ratings) to tune the 0.7 / 0.2 / 0.1 weights.
- Add collaborative filtering for a hybrid recommender once user histories are available.
- Replace VADER with a transformer sentiment model.
- Deploy the app and expose search through an API.

## Team

- **Abhishek Teotia:** recommendation engine, retrieval pipeline, integration and testing
- **Muskan Choudhary:** NLP query extraction
- **Justin Govanxa:** sentiment module
- **Yuzhong Wang:** front end and LLM integration

For educational and research use.
