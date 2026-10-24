from movie_recommendation_system.recommender import parse_genres


def test_similar_excludes_seed_and_ranks_same_cluster_first(recommender):
    seed, results = recommender.similar("john wick", k=2)
    assert seed.title == "John Wick"
    titles = [r.title for r in results]
    assert "John Wick" not in titles
    assert set(titles) <= {"John Wick: Chapter 2", "Taken"}


def test_find_title_handles_typos_and_rejects_noise(recommender):
    assert recommender.movies.title[recommender.find_title("toy stroy")] == "Toy Story"
    assert recommender.find_title("xqzv blorp") is None


def test_search_uses_embedder(recommender):
    results = recommender.search("an animated film for kids", k=3)
    assert {r.title for r in results} == {"Toy Story", "Toy Story 2", "Finding Nemo"}


def test_browse_filters(recommender):
    view = recommender.browse(genres=["animation"], years=(1990, 2000))
    assert set(view.title) == {"Toy Story", "Toy Story 2"}
    assert list(recommender.browse(title_contains="wick").title) == ["John Wick: Chapter 2", "John Wick"]


def test_parse_genres():
    assert parse_genres("['Comedy', 'Crime']") == ["Comedy", "Crime"]
    assert parse_genres("Comedy, Crime") == ["Comedy", "Crime"]
    assert parse_genres(None) == []
