"""HTTP API over the recommender."""

from __future__ import annotations

import os
from dataclasses import asdict
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from movie_recommendation_system import __version__
from movie_recommendation_system.recommender import DEFAULT_ARTIFACTS, Recommender

app = FastAPI(title="Movie Recommendation API", version=__version__)


class Movie(BaseModel):
    index: int
    title: str
    year: int | None
    genres: list[str]
    overview: str
    poster_path: str | None
    similarity: float
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[Movie]


class SimilarResponse(BaseModel):
    seed: Movie
    results: list[Movie]


@lru_cache
def get_recommender() -> Recommender:
    return Recommender.from_artifacts(os.getenv("ARTIFACTS_DIR", DEFAULT_ARTIFACTS))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "movies": len(get_recommender().movies)}


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=2, max_length=500, description="What you want to watch, in plain English"),
    k: int = Query(12, ge=1, le=50),
) -> SearchResponse:
    results = get_recommender().search(q, k)
    return SearchResponse(query=q, results=[Movie(**asdict(r)) for r in results])


@app.get("/similar", response_model=SimilarResponse)
def similar(
    title: str = Query(..., min_length=1, max_length=200),
    k: int = Query(12, ge=1, le=50),
) -> SimilarResponse:
    found = get_recommender().similar(title, k)
    if found is None:
        raise HTTPException(404, f"No catalogue title closely matches {title!r}")
    seed, results = found
    return SimilarResponse(seed=Movie(**asdict(seed)), results=[Movie(**asdict(r)) for r in results])


@app.get("/genres")
def genres() -> list[str]:
    return get_recommender().genre_vocabulary
