"""Streamlit front end. All search and ranking logic lives in movie_recommendation_system."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `streamlit run App/app.py` works without installing the package

from movie_recommendation_system.assistant import candidate_titles, load_llm, load_spacy, summarise
from movie_recommendation_system.recommender import Recommendation, Recommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
FALLBACK_POSTER = Path(__file__).parent / "static" / "no_poster.png"
PAGE_SIZE = 25


@st.cache_resource(show_spinner="Loading 44,471 movies and the search index…")
def get_recommender() -> Recommender:
    return Recommender.from_artifacts(ROOT / "artifacts")


@st.cache_resource(show_spinner=False)
def get_helpers():
    return load_llm(), load_spacy()


@st.cache_data(show_spinner=False, max_entries=2000)
def poster(path: str | None, size: str = "w342"):
    if path:
        try:
            r = requests.get(f"https://image.tmdb.org/t/p/{size}{path}", timeout=3)
            if r.ok and r.headers.get("Content-Type", "").startswith("image/"):
                return Image.open(BytesIO(r.content))
        except requests.RequestException:
            pass
    return Image.open(FALLBACK_POSTER)


def show_grid(results: list[Recommendation], columns: int = 4) -> None:
    cols = st.columns(columns)
    for i, rec in enumerate(results):
        with cols[i % columns]:
            st.image(poster(rec.poster_path), use_container_width=True)
            year = f" ({rec.year})" if rec.year else ""
            st.markdown(f"**{i + 1}. {rec.title}**{year}")
            st.caption(" · ".join(rec.genres[:3]))
            with st.expander("Overview"):
                st.write(rec.overview or "No overview available.")


rec = get_recommender()
llm, nlp = get_helpers()

home, browse, similar = st.tabs(["🏠 Ask", "📚 Browse", "🎯 Similar by title"])

with home:
    st.title("🎬 What do you feel like watching?")
    st.caption(
        'Describe a mood or plot ("a slow-burn mystery set in a small town"), or name a film you liked.'
    )
    prompt = st.chat_input("Tell me what you're in the mood for…")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            seed = next(
                (i for t in candidate_titles(prompt, llm, nlp) if (i := rec.find_title(t)) is not None), None
            )
            if seed is not None:
                seed_title = rec.movies.title[seed]
                st.markdown(f"Movies similar to **{seed_title}**:")
                _, results = rec.similar(seed_title)
            else:
                with st.spinner("Searching by meaning…"):
                    results = rec.search(prompt)
            st.caption(summarise(results, rec.sentiment, llm))
            show_grid(results)

with browse:
    st.title("📚 Browse the catalogue")
    years = rec.movies["year"].dropna()
    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    year_range = c1.slider("Year", int(years.min()), int(years.max()), (1980, int(years.max())))
    genres = c2.multiselect("Genres", rec.genre_vocabulary)
    title_query = c3.text_input("Title contains")
    sort_by = c4.selectbox("Sort by", ["popularity", "year", "title"])
    view = rec.browse(genres, year_range, title_query, sort_by)

    pages = max(1, -(-len(view) // PAGE_SIZE))
    page = st.number_input(f"Page (of {pages})", min_value=1, max_value=pages, value=1)
    start = (page - 1) * PAGE_SIZE
    st.caption(f"{len(view):,} movies · showing {start + 1:,}–{min(start + PAGE_SIZE, len(view)):,}")
    cols = st.columns(5)
    for j, (_, row) in enumerate(view.iloc[start : start + PAGE_SIZE].iterrows()):
        with cols[j % 5]:
            st.image(poster(row.get("poster_path") if isinstance(row.get("poster_path"), str) else None))
            year = f" ({int(row['year'])})" if row["year"] == row["year"] else ""
            st.markdown(f"**{row['title']}**{year}")

with similar:
    st.title("🎯 Find similar movies")
    title = st.text_input("A movie you like", placeholder="e.g. John Wick")
    if title:
        found = rec.similar(title)
        if found is None:
            st.warning("No close match in the catalogue. Check the spelling or try another title.")
        else:
            seed, results = found
            left, right = st.columns([1, 3])
            left.image(poster(seed.poster_path), use_container_width=True)
            right.subheader(f"{seed.title} ({seed.year})" if seed.year else seed.title)
            right.write(seed.overview)
            st.caption(summarise(results, rec.sentiment, llm))
            show_grid(results)

st.sidebar.markdown(
    "**How it works**\n\n"
    "Each movie's overview, genres and review snippets are embedded with SBERT (all-MiniLM-L6-v2). "
    "Queries retrieve 200 neighbours from an HNSW index, which are ranked by similarity (70%), "
    "review sentiment (20%) and popularity (10%), then diversified with MMR.\n\n"
    f"Catalogue: {len(rec.movies):,} movies"
)
