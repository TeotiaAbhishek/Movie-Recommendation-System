"""A tiny in-memory catalogue so tests need neither the 200 MB artifacts nor a model download."""

import numpy as np
import pandas as pd
import pytest

from movie_recommendation_system.recommender import Recommender, Settings

TITLES = [
    ("John Wick", "['Action', 'Thriller']", 0),
    ("John Wick: Chapter 2", "['Action', 'Thriller']", 0),
    ("Taken", "['Action', 'Thriller']", 0),
    ("Toy Story", "['Animation', 'Family']", 1),
    ("Toy Story 2", "['Animation', 'Family']", 1),
    ("Finding Nemo", "['Animation', 'Family']", 1),
    ("Pride and Prejudice", "['Drama', 'Romance']", 2),
    ("Emma", "['Drama', 'Romance']", 2),
]


class FakeIndex:
    """Exact cosine search with hnswlib's knn_query interface."""

    def __init__(self, vectors):
        self.vectors = vectors

    def knn_query(self, query, k):
        sims = self.vectors @ query[0]
        order = np.argsort(-sims)[:k]
        return order[None, :], (1.0 - sims[order])[None, :]


class FakeEmbedder:
    """Maps a query to the centre of a cluster by keyword."""

    def __init__(self, centres):
        self.centres = centres

    def encode(self, texts, normalize_embeddings=True):
        text = texts[0].lower()
        cluster = 1 if "animat" in text or "kids" in text else 2 if "romance" in text else 0
        return self.centres[cluster : cluster + 1]


@pytest.fixture
def recommender():
    rng = np.random.default_rng(0)
    centres = np.eye(3, 16, dtype="float32")
    vectors = np.stack([centres[c] + 0.15 * rng.standard_normal(16) for _, _, c in TITLES]).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    movies = pd.DataFrame(
        {
            "title": [t for t, _, _ in TITLES],
            "genres": [g for _, g, _ in TITLES],
            "year": [2014, 2017, 2008, 1995, 1999, 2003, 2005, 1996],
            "overview": ["..."] * len(TITLES),
            "poster_path": [None] * len(TITLES),
            "popularity": np.linspace(1, 50, len(TITLES)),
            "vote_count": [1000] * len(TITLES),
        }
    )
    sentiment = np.full(len(TITLES), 0.5, dtype="float32")
    return Recommender(
        movies, vectors, FakeIndex(vectors), sentiment, FakeEmbedder(centres), Settings(candidates=7)
    )
