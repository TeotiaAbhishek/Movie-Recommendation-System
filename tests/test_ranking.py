import numpy as np

from movie_recommendation_system.ranking import fuse_scores, mmr, normalize01


def test_normalize01_range_and_constant():
    assert np.allclose(normalize01(np.array([2.0, 4.0, 6.0])), [0.0, 0.5, 1.0])
    assert np.allclose(normalize01(np.array([3.0, 3.0])), [0.0, 0.0])


def test_fuse_scores_weights():
    sims = np.array([0.9, 0.5, 0.1])
    sent = np.array([0.0, 1.0, 0.0])
    only_sim = fuse_scores(sims, sent, None, 1.0, 0.0, 0.0)
    assert np.argmax(only_sim) == 0
    sentiment_heavy = fuse_scores(sims, sent, None, 0.1, 0.9, 0.0)
    assert np.argmax(sentiment_heavy) == 1


def test_mmr_prefers_diverse_second_pick():
    # Candidates 0 and 1 are near-duplicates; 2 is different but slightly less relevant.
    vectors = np.array([[1.0, 0.0], [0.999, 0.045], [0.0, 1.0]], dtype="float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = np.array([1.0, 0.95, 0.9], dtype="float32")
    assert mmr(vectors, scores, k=2, lambda_=0.5) == [0, 2]
    assert mmr(vectors, scores, k=2, lambda_=1.0) == [0, 1]  # pure relevance


def test_mmr_returns_each_candidate_once():
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((20, 4)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    picks = mmr(vectors, rng.random(20).astype("float32"), k=20, lambda_=0.7)
    assert sorted(picks) == list(range(20))
