"""Score fusion and diversity re-ranking, independent of how candidates were retrieved."""

from __future__ import annotations

import numpy as np


def normalize01(x: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1]; a constant vector becomes all zeros."""
    x = np.asarray(x, dtype="float32")
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    return np.zeros_like(x) if hi <= lo else (x - lo) / (hi - lo)


def fuse_scores(
    similarity: np.ndarray,
    sentiment: np.ndarray | None,
    popularity: np.ndarray | None,
    w_sim: float,
    w_sent: float,
    w_pop: float,
) -> np.ndarray:
    """Weighted sum of similarity, sentiment (already 0-1) and popularity, each scaled to 0-1."""
    score = w_sim * normalize01(similarity)
    if sentiment is not None:
        score = score + w_sent * np.asarray(sentiment, dtype="float32")
    if popularity is not None:
        score = score + w_pop * normalize01(np.log1p(np.nan_to_num(popularity)))
    return score


def mmr(vectors: np.ndarray, scores: np.ndarray, k: int, lambda_: float) -> list[int]:
    """Maximal Marginal Relevance: pick ``k`` positions trading relevance against redundancy.

    ``vectors`` are L2-normalised candidate embeddings, so a dot product is cosine similarity.
    Returns positions into ``vectors``/``scores``.
    """
    n = len(scores)
    k = min(k, n)
    selected: list[int] = []
    # Highest similarity of each candidate to anything selected so far.
    max_sim_to_selected = np.full(n, -np.inf, dtype="float32")
    available = np.ones(n, dtype=bool)
    for _ in range(k):
        redundancy = np.where(np.isfinite(max_sim_to_selected), max_sim_to_selected, 0.0)
        mmr_score = lambda_ * scores - (1.0 - lambda_) * redundancy
        mmr_score[~available] = -np.inf
        best = int(np.argmax(mmr_score))
        selected.append(best)
        available[best] = False
        max_sim_to_selected = np.maximum(max_sim_to_selected, vectors @ vectors[best])
    return selected
