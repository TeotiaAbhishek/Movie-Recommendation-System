"""Offline evaluation of similar-title recommendations.

There are no user ratings in the data, so these are proxy metrics computed over
seed movies drawn from the catalogue:

- genre precision@k: share of recommendations sharing at least one genre with the seed
- diversity: 1 - mean pairwise cosine similarity within each list (higher = less repetitive)
- coverage: share of the catalogue that appears in any list
- latency: time per query, p50 and p95
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np

from movie_recommendation_system.recommender import Recommender, Settings

CONFIGS = {
    "similarity only": Settings(w_sim=1.0, w_sent=0.0, w_pop=0.0, use_mmr=False),
    "fusion, no MMR": Settings(use_mmr=False),
    "fusion + MMR (default)": Settings(),
}


def evaluate(recommender: Recommender, seeds: list[int], k: int = 12) -> dict:
    precision, diversity, latency = [], [], []
    seen: set[int] = set()
    for seed in seeds:
        start = time.perf_counter()
        recs = recommender._rank(recommender.embeddings[seed : seed + 1], k, exclude=seed)
        latency.append(time.perf_counter() - start)
        seed_genres = set(recommender.genre_lists.iloc[seed])
        ids = [r.index for r in recs]
        seen.update(ids)
        precision.append(np.mean([bool(seed_genres & set(r.genres)) for r in recs]))
        vectors = recommender.embeddings[ids]
        sims = vectors @ vectors.T
        n = len(ids)
        diversity.append(1.0 - (sims.sum() - n) / (n * (n - 1)))
    return {
        "genre_precision_at_k": round(float(np.mean(precision)), 3),
        "diversity": round(float(np.mean(diversity)), 3),
        "coverage": round(len(seen) / len(recommender.movies), 4),
        "latency_ms_p50": round(float(np.percentile(latency, 50)) * 1000, 1),
        "latency_ms_p95": round(float(np.percentile(latency, 95)) * 1000, 1),
    }


def pick_seeds(recommender: Recommender, n: int, min_votes: int, seed: int) -> list[int]:
    """Random well-known movies (enough votes) that have at least one genre."""
    movies = recommender.movies
    eligible = movies.index[(movies["vote_count"] >= min_votes) & recommender.genre_lists.map(bool)]
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(eligible, size=min(n, len(eligible)), replace=False).tolist())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=500)
    parser.add_argument("--min-votes", type=int, default=100)
    parser.add_argument("--k", type=int, default=12)
    args = parser.parse_args(argv)

    # Similar-title queries reuse stored embeddings, so the text model is never loaded.
    recommender = Recommender.from_artifacts()
    seeds = pick_seeds(recommender, args.seeds, args.min_votes, seed=42)
    results = {}
    for name, settings in CONFIGS.items():
        recommender.settings = replace(settings)
        results[name] = evaluate(recommender, seeds, args.k)
    print(json.dumps({"seeds": len(seeds), "k": args.k, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
