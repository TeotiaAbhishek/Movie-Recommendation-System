import pytest
from fastapi.testclient import TestClient

from movie_recommendation_system import api


@pytest.fixture
def client(recommender, monkeypatch):
    monkeypatch.setattr(api, "get_recommender", lambda: recommender)
    return TestClient(api.app)


def test_health(client):
    assert client.get("/health").json()["movies"] == 8


def test_search(client):
    body = client.get("/search", params={"q": "a romance", "k": 2}).json()
    assert {m["title"] for m in body["results"]} == {"Pride and Prejudice", "Emma"}


def test_similar(client):
    body = client.get("/similar", params={"title": "Toy Story", "k": 2}).json()
    assert body["seed"]["title"] == "Toy Story"
    assert len(body["results"]) == 2


def test_similar_unknown_title_is_404(client):
    assert client.get("/similar", params={"title": "xqzv blorp"}).status_code == 404


def test_validation(client):
    assert client.get("/search", params={"q": "x"}).status_code == 422
    assert client.get("/search", params={"q": "funny", "k": 500}).status_code == 422
