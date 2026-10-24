"""Load the prebuilt artifacts and answer free-text, similar-title and browse queries."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process, utils

from movie_recommendation_system.ranking import fuse_scores, mmr

DEFAULT_ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


class Embedder(Protocol):
    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray: ...


@dataclass(frozen=True)
class Settings:
    w_sim: float = 0.70
    w_sent: float = 0.20
    w_pop: float = 0.10
    use_mmr: bool = True
    mmr_lambda: float = 0.70
    candidates: int = 200
    title_match_threshold: int = 80


@dataclass
class Recommendation:
    index: int
    title: str
    year: int | None
    genres: list[str]
    overview: str
    poster_path: str | None
    similarity: float
    score: float


@dataclass
class Recommender:
    movies: pd.DataFrame
    embeddings: np.ndarray
    index: object  # hnswlib.Index, or anything with knn_query(vectors, k)
    sentiment: np.ndarray
    embedder: Embedder | None = None
    settings: Settings = field(default_factory=Settings)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @classmethod
    def from_artifacts(
        cls,
        path: str | Path = DEFAULT_ARTIFACTS,
        embedder: Embedder | None = None,
        settings: Settings | None = None,
    ) -> Recommender:
        import hnswlib

        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        movies = pd.read_parquet(path / "movies_aug.parquet").reset_index(drop=True)
        embeddings = np.load(path / "embeddings.npy").astype("float32")
        index = hnswlib.Index(space="cosine", dim=embeddings.shape[1])
        index.load_index(str(path / "hnsw_cosine.idx"), max_elements=len(embeddings))
        index.set_ef(200)
        sentiment_path = path / "review_sentiment_01.npy"
        sentiment = np.load(sentiment_path).astype("float32") if sentiment_path.exists() else None
        if sentiment is None or len(sentiment) != len(movies):
            sentiment = np.zeros(len(movies), dtype="float32")
        return cls(
            movies, embeddings, index, sentiment, embedder, settings or Settings(), config["embedding_model"]
        )

    # ---------------------------------------------------------------- queries

    def search(self, query: str, k: int = 12) -> list[Recommendation]:
        """Recommend movies matching a free-text description."""
        if self.embedder is None:
            # Loaded on first use: title search and browsing work without the model.
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(self.embedding_model)
        vector = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        return self._rank(vector, k)

    def similar(self, title: str, k: int = 12) -> tuple[Recommendation, list[Recommendation]] | None:
        """Movies similar to the best catalogue match for ``title``; None if nothing matches well."""
        seed = self.find_title(title)
        if seed is None:
            return None
        vector = self.embeddings[seed : seed + 1]
        seed_rec = self._recommendation(seed, similarity=1.0, score=1.0)
        return seed_rec, self._rank(vector, k, exclude=seed)

    def find_title(self, text: str) -> int | None:
        """Row index of the closest catalogue title, or None below the match threshold."""
        match = process.extractOne(
            text, self._titles, scorer=fuzz.token_sort_ratio, processor=utils.default_process
        )
        if match is None or match[1] < self.settings.title_match_threshold:
            return None
        return int(match[2])

    def browse(
        self,
        genres: list[str] | None = None,
        years: tuple[int, int] | None = None,
        title_contains: str = "",
        sort_by: str = "popularity",
    ) -> pd.DataFrame:
        view = self.movies
        if years:
            view = view[view["year"].between(*years)]
        if genres:
            wanted = {g.lower() for g in genres}
            view = view[[wanted <= {g.lower() for g in row} for row in self.genre_lists[view.index]]]
        if title_contains.strip():
            view = view[view["title"].str.contains(title_contains.strip(), case=False, na=False, regex=False)]
        return view.sort_values(sort_by, ascending=sort_by != "popularity", na_position="last")

    # ------------------------------------------------------------- internals

    @cached_property
    def _titles(self) -> list[str]:
        return self.movies["title"].fillna("").tolist()

    @cached_property
    def genre_lists(self) -> pd.Series:
        return self.movies["genres"].map(parse_genres)

    @cached_property
    def genre_vocabulary(self) -> list[str]:
        return sorted({g for row in self.genre_lists for g in row})

    def _rank(self, vector: np.ndarray, k: int, exclude: int | None = None) -> list[Recommendation]:
        s = self.settings
        n_candidates = min(s.candidates + 1, len(self.embeddings))
        labels, distances = self.index.knn_query(vector, k=n_candidates)
        ids, sims = labels[0].astype(int), 1.0 - distances[0]
        keep = ids != exclude
        ids, sims = ids[keep], sims[keep]
        popularity = self.movies["popularity"].to_numpy()[ids] if "popularity" in self.movies else None
        scores = fuse_scores(sims, self.sentiment[ids], popularity, s.w_sim, s.w_sent, s.w_pop)
        if s.use_mmr:
            order = mmr(self.embeddings[ids], scores, k, s.mmr_lambda)
        else:
            order = list(np.argsort(-scores)[:k])
        return [self._recommendation(int(ids[i]), float(sims[i]), float(scores[i])) for i in order]

    def _recommendation(self, i: int, similarity: float, score: float) -> Recommendation:
        row = self.movies.iloc[i]
        year = row.get("year")
        return Recommendation(
            index=i,
            title=str(row.get("title") or ""),
            year=int(year) if pd.notna(year) else None,
            genres=self.genre_lists.iloc[i],
            overview=str(row.get("overview") or ""),
            poster_path=row.get("poster_path") if isinstance(row.get("poster_path"), str) else None,
            similarity=round(similarity, 4),
            score=round(score, 4),
        )


def parse_genres(cell) -> list[str]:
    """Genres are stored as a stringified list ("['Comedy', 'Crime']"); accept plain lists too."""
    if isinstance(cell, list):
        return [str(g).strip() for g in cell if str(g).strip()]
    if isinstance(cell, str) and cell.strip():
        try:
            value = ast.literal_eval(cell)
            if isinstance(value, list):
                return [str(g).strip() for g in value if str(g).strip()]
        except (ValueError, SyntaxError):
            return [g.strip() for g in cell.split(",") if g.strip()]
    return []
