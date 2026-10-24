"""Optional natural-language helpers: pull movie titles out of a request and summarise results.

A local LLM (llama-cpp, GGUF model at LLM_MODEL_PATH) is used when available. Without one,
title extraction falls back to quoted phrases and spaCy WORK_OF_ART entities, and the summary
is rule-based, so the app works fully offline.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

from movie_recommendation_system.recommender import Recommendation


def load_llm():
    model_path = os.getenv("LLM_MODEL_PATH", "").strip()
    if not model_path or not Path(model_path).exists():
        return None
    try:
        from llama_cpp import Llama
    except ImportError:
        return None
    return Llama(model_path=model_path, n_ctx=2048, n_threads=os.cpu_count() or 4, verbose=False)


def load_spacy():
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        return None


def _chat(llm, system: str, user: str, max_tokens: int) -> str | None:
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return out["choices"][0]["message"]["content"].strip()


def candidate_titles(prompt: str, llm=None, nlp=None) -> list[str]:
    """Phrases in the request that may be movie titles, most likely first."""
    if llm is not None:
        reply = _chat(
            llm,
            "Extract movie titles from the user text. Reply with a JSON array of strings only.",
            prompt,
            max_tokens=128,
        )
        try:
            titles = json.loads(reply or "[]")
            if isinstance(titles, list) and titles:
                return [str(t) for t in titles]
        except json.JSONDecodeError:
            pass
    found = [q.strip() for q in re.findall(r"[\"“”']([^\"“”']{2,})[\"“”']", prompt)]
    if nlp is not None:
        found += [ent.text.strip() for ent in nlp(prompt).ents if ent.label_ == "WORK_OF_ART"]
    m = re.search(r"\b(?:like|similar to)\s+(.+)$", prompt, flags=re.IGNORECASE)
    if m:
        found.append(m.group(1).strip(" .!?"))
    return found or [prompt]


def summarise(results: list[Recommendation], sentiment: np.ndarray, llm=None) -> str:
    if not results:
        return "No results."
    if llm is not None:
        rows = [{"title": r.title, "year": r.year, "genres": r.genres} for r in results]
        reply = _chat(
            llm, "Summarise these movie picks in one sentence, naming shared themes.", str(rows), 80
        )
        if reply:
            return reply
    genres = Counter(g for r in results for g in r.genres)
    top = ", ".join(g for g, _ in genres.most_common(3)) or "varied genres"
    years = [r.year for r in results if r.year]
    span = f"{min(years)}–{max(years)}" if years else "various years"
    mood = float(np.mean(sentiment[[r.index for r in results]]))
    tone = "very positive" if mood > 0.7 else "mostly positive" if mood > 0.55 else "mixed"
    return f"Leaning towards **{top}**, from **{span}**, with **{tone}** audience reviews."
