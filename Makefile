.PHONY: install test lint format app api evaluate docker

install:
	pip install -e ".[app,dev]"

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

app:
	streamlit run App/app.py

api:
	uvicorn movie_recommendation_system.api:app --reload

evaluate:
	python -m movie_recommendation_system.evaluate

docker:
	docker build -t movie-recommender . && docker run --rm -p 8000:8000 movie-recommender
