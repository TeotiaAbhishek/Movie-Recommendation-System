# app.py
from __future__ import annotations
import json, os, re, base64, requests
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import hnswlib
from sentence_transformers import SentenceTransformer
from rapidfuzz import process, fuzz
from io import BytesIO
from PIL import Image

try:
    from llama_cpp import Llama
except Exception:
    Llama = None
try:
    import spacy
    _HAS_SPACY = True
except Exception:
    _HAS_SPACY = False


# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
ART = Path("../artifacts")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

W_SIM, W_SENT, W_POP = 0.70, 0.20, 0.10
USE_MMR, MMR_LAMBDA = True, 0.70
TOPK_CAND, TOPK_SHOW = 200, 12
BROWSE_PAGE_SIZE = 50
TITLE_CONF_OK = 80


# ---------------- Load Data ----------------
@st.cache_data(show_spinner=False)
def load_movies() -> pd.DataFrame:
    df = pd.read_parquet(ART / "movies_aug.parquet")
    for c in ["title","overview","genres","genres_text","year","popularity","poster_path","reviews_text"]:
        if c not in df.columns: df[c] = None
    if df["year"].isna().all() and "release_date" in df.columns:
        df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    return df.reset_index(drop=True)

@st.cache_resource(show_spinner=False)
def load_embeddings() -> np.ndarray:
    emb = np.load(ART / "embeddings.npy")
    return emb.astype("float32")

@st.cache_data(show_spinner=False)
def load_sentiment01(n_rows: int) -> np.ndarray:
    p = ART / "review_sentiment_01.npy"
    if p.exists():
        x = np.load(p).astype("float32")
        if x.shape[0] == n_rows:
            return x
    return np.zeros(n_rows, dtype="float32")

@st.cache_resource(show_spinner=True)
def load_index(emb: np.ndarray) -> hnswlib.Index:
    dim = emb.shape[1]
    idx = hnswlib.Index(space="cosine", dim=dim)
    path = ART / "hnsw_cosine.idx"
    if path.exists():
        idx.load_index(str(path), max_elements=emb.shape[0])
        idx.set_ef(200)
    else:
        idx.init_index(max_elements=emb.shape[0], ef_construction=200, M=16)
        idx.add_items(emb, np.arange(emb.shape[0]))
        idx.set_ef(200)
    return idx

@st.cache_resource(show_spinner=False)
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMB_MODEL)

@st.cache_data(show_spinner=False)
def load_id_map() -> Dict:
    with open(ART / "id_map.json") as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def load_spacy() -> "spacy.language.Language|None":
    if not _HAS_SPACY: return None
    return spacy.load("en_core_web_sm")

@st.cache_resource(show_spinner=False)
def load_llm() -> "Llama|None":
    model_path = os.getenv("LLM_MODEL_PATH", "").strip()
    if not (Llama and model_path and Path(model_path).exists()):
        return None
    return Llama(model_path=model_path, n_ctx=2048, n_threads=os.cpu_count() or 4)


# ---------------- Poster Handling ----------------
@st.cache_data(show_spinner=False)
def get_fallback() -> str:
    fallback_path = Path(__file__).parent / "static" / "no_poster.png"
    return Image.open(fallback_path)

def tmdb_poster_url(poster_path: str | None, size: str = "w342") -> str:
    if isinstance(poster_path, str) and poster_path.strip():
        url = f"https://image.tmdb.org/t/p/{size}{poster_path}"
        r = requests.get(url, timeout=2)
        content_type = r.headers.get("Content-Type", "")
        if r.status_code == 200 and any(ext in content_type for ext in ["jpeg", "jpg", "png", "webp"]):
            return Image.open(BytesIO(r.content))
    return get_fallback()

# ---------------- Utility Functions ----------------
def parse_genres_cell(x) -> List[str]:
    if isinstance(x, list):
        if x and isinstance(x[0], dict):
            return [d.get("name","").strip() for d in x if isinstance(d,dict)]
        return [str(t).strip() for t in x if str(t).strip()]
    if isinstance(x, str):
        s = x.strip()
        if "," in s: return [t.strip() for t in s.split(",") if t.strip()]
        return [t.strip() for t in s.split() if t.strip()]
    return []

def knn_query(index: hnswlib.Index, qv: np.ndarray, k: int) -> Tuple[np.ndarray,np.ndarray]:
    labels, distances = index.knn_query(qv.astype("float32"), k=k)
    idxs = labels[0]
    sims = 1.0 - distances[0]
    return idxs, sims

def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
    return np.zeros_like(x) if mx <= mn else (x - mn) / (mx - mn + 1e-9)

def fuse_scores(sim: np.ndarray, ids: np.ndarray, sent01: np.ndarray|None,
                pop_vec: np.ndarray|None, w_sim=W_SIM, w_sent=W_SENT, w_pop=W_POP) -> np.ndarray:
    parts = [w_sim * normalize01(sim)]
    if sent01 is not None: parts.append(w_sent * sent01[ids].astype("float32"))
    if pop_vec is not None: parts.append(w_pop * normalize01(pop_vec))
    return np.sum(np.vstack(parts), axis=0)

def mmr(q_vec: np.ndarray, cand_vecs: np.ndarray, cand_order: np.ndarray,
        base_scores: np.ndarray, lambda_=MMR_LAMBDA, k: int = TOPK_SHOW) -> List[int]:
    selected, pool = [], list(range(len(cand_order)))
    for _ in range(min(k, len(pool))):
        scores = []
        for i in pool:
            rel = float(base_scores[i])
            div = float(np.max(cand_vecs[i] @ cand_vecs[selected].T)) if selected else 0.0
            scores.append(lambda_ * rel - (1.0 - lambda_) * div)
        j = pool[int(np.argmax(scores))]
        selected.append(j); pool.remove(j)
    return [int(cand_order[j]) for j in selected]


# ---------------- Offline LLM utils ----------------
def llm_chat(llm: "Llama|None", system: str, user: str, temperature: float = 0.2, max_tokens: int = 256) -> str|None:
    if llm is None: return None
    out = llm.create_chat_completion(
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=temperature, max_tokens=max_tokens
    )
    return out["choices"][0]["message"]["content"].strip()

def extract_titles_from_prompt(prompt: str, catalog_titles: List[str], llm: "Llama|None", nlp) -> List[Tuple[str,int]]:
    if llm:
        sys = "Extract movie titles from the user text. Return a pure JSON array of strings (titles), nothing else."
        usr = f"Text: {prompt}\nOutput: a JSON array of movie titles mentioned or clearly implied."
        txt = llm_chat(llm, sys, usr, temperature=0.0, max_tokens=128)
        if txt:
            try:
                import json as _json
                titles = _json.loads(txt)
                if isinstance(titles, list) and titles:
                    hits = []
                    for t in titles:
                        m = process.extractOne(str(t), catalog_titles, scorer=fuzz.WRatio)
                        if m: hits.append((m[0], int(m[1])))
                    best = {}
                    for t,s in hits:
                        if t not in best or s>best[t]: best[t]=s
                    return sorted(best.items(), key=lambda x:x[1], reverse=True)
            except Exception:
                pass
    candidates = set([q.strip() for q in re.findall(r"[\"“”']([^\"“”']{2,})[\"“”']", prompt)])
    if nlp:
        doc = nlp(prompt)
        for ent in doc.ents:
            if ent.label_ in ("WORK_OF_ART",): candidates.add(ent.text.strip())
    if candidates:
        hits = []
        for c in candidates:
            m = process.extractOne(c, catalog_titles, scorer=fuzz.WRatio)
            if m: hits.append((m[0], int(m[1])))
        best = {}
        for t,s in hits:
            if t not in best or s>best[t]: best[t]=s
        return sorted(best.items(), key=lambda x:x[1], reverse=True)
    res = process.extract(prompt, catalog_titles, scorer=fuzz.WRatio, limit=5)
    return [(r[0], int(r[1])) for r in res]

# ---------------- Summarization (LLM or fallback) ----------------
def summarize_results_llm(results_df: pd.DataFrame, llm: "Llama|None", sent: np.ndarray|None) -> str:
    if results_df.empty:
        return "No results to summarize."

    if llm is None:
        genre_counts = {}
        for x in results_df.get("genres_text", results_df.get("genres", "")).fillna(""):
            for g in parse_genres_cell(x):
                if g:
                    genre_counts[g] = genre_counts.get(g, 0) + 1

        top_genres = ", ".join(
            [g for g, _ in sorted(genre_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]]
        ) or "varied"

        yrs = results_df["year"].dropna().astype(int)
        year_span = f"{int(yrs.min())}–{int(yrs.max())}" if not yrs.empty else "varied years"

        if sent is not None and len(sent) > 0:
            m = float(np.nanmean(sent))
            mood = (
                "very positive" if m > 0.7
                else "mixed/positive" if m > 0.55
                else "mixed"
            )
        else:
            mood = "mixed"

        return f"Leaning toward **{top_genres}**, across **{year_span}**, with overall **{mood}** reviews."

    rows = results_df[["title", "year", "genres_text", "overview"]].fillna("").head(12).to_dict(orient="records")
    sys = "You are a concise movie critic. Write a 1–2 sentence summary of these movies, mentioning genres or themes."
    usr = f"Movies: {rows}\nOptional sentiment average: {float(np.nanmean(sent)) if sent is not None and len(sent) > 0 else 'n/a'}"
    out = llm_chat(llm, sys, usr, temperature=0.2, max_tokens=120)
    return out or "Here are some top matches based on your preferences."



# ---------------- Load Artifacts ----------------
movies = load_movies()
emb = load_embeddings()
index = load_index(emb)
sent01 = load_sentiment01(len(movies))
embedder = load_embedder()
id_map = load_id_map()
nlp = load_spacy()
llm = load_llm()
titles = movies["title"].fillna("").tolist()


# ---------------- UI Tabs ----------------
st.markdown(
    """
    <style>
    .topnav {display:flex; gap:1rem; align-items:center; padding:0.5rem 0;}
    .topnav a {padding:.4rem .8rem; border-radius:.5rem; text-decoration:none; background:#1f1f1f; color:#fff;}
    .topnav a.active {background:#ff4b4b;}
    </style>
    """,
    unsafe_allow_html=True,
)
tabs = st.tabs(["🏠 Home", "📚 Browse", "🎯 Similar by Title"])

# ============================== HOME ===============================
with tabs[0]:
    st.title("🎬 Movie Recommender — Chat")
    st.caption("Type a vibe or name some films. (Offline LLM tries to extract titles; otherwise fuzzy match.)")

    if "chat" not in st.session_state: st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Tell me what you’re in the mood for…")
    if prompt:
        st.session_state.chat.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)

        hits = extract_titles_from_prompt(prompt, titles, llm, nlp)
        seed_title = hits[0][0] if (hits and hits[0][1] >= TITLE_CONF_OK) else None

        with st.chat_message("assistant"):
            if seed_title:
                st.markdown(f"Finding movies similar to **{seed_title}**…")
                seed_rows = movies[movies["title"] == seed_title]
                if seed_rows.empty:
                    st.warning("Matched title not found in table.")
                else:
                    seed_row = seed_rows.iloc[0]
                    seed_idx = int(seed_row.name)
                    seed_vec = emb[seed_idx:seed_idx+1]
                    idxs, sims = knn_query(index, seed_vec, k=TOPK_CAND)
                    if len(idxs) and idxs[0] == seed_idx:
                        idxs, sims = idxs[1:], sims[1:]
                    pops = movies.loc[idxs, "popularity"].values if "popularity" in movies.columns else None
                    fused = fuse_scores(sims, idxs, sent01, pops, W_SIM, W_SENT, W_POP)
                    order = mmr(seed_vec[0], emb[idxs], np.arange(len(idxs)), fused, lambda_=MMR_LAMBDA, k=TOPK_SHOW) if USE_MMR else np.argsort(-fused)[:TOPK_SHOW]
                    final_idxs = idxs[order]; final_sims = sims[order]
                    recs = movies.iloc[final_idxs].copy()
                    st.caption(summarize_results_llm(recs, llm, sent01[final_idxs] if len(final_idxs)>0 else None))
                    for rank, (_, row) in enumerate(recs.iterrows(), start=1):
                        st.markdown(f"### {rank}. {row['title']} ({int(row['year']) if pd.notna(row['year']) else '—'})")
                        st.image(tmdb_poster_url(row.get("poster_path")), use_container_width=False)
                        st.write(str(row.get("overview","") or ""))
            else:
                qv = embedder.encode([prompt], normalize_embeddings=True).astype("float32")
                idxs, sims = knn_query(index, qv, k=TOPK_CAND)
                pops = movies.loc[idxs, "popularity"].values if "popularity" in movies.columns else None
                fused = fuse_scores(sims, idxs, sent01, pops, W_SIM, W_SENT, W_POP)
                order = mmr(qv[0], emb[idxs], np.arange(len(idxs)), fused, lambda_=MMR_LAMBDA, k=TOPK_SHOW) if USE_MMR else np.argsort(-fused)[:TOPK_SHOW]
                final_idxs = idxs[order]; final_sims = sims[order]
                recs = movies.iloc[final_idxs].copy()
                st.caption(summarize_results_llm(recs, llm, sent01[final_idxs] if len(final_idxs)>0 else None))
                for rank, (_, row) in enumerate(recs.iterrows(), start=1):
                    st.markdown(f"### {rank}. {row['title']} ({int(row['year']) if pd.notna(row['year']) else '—'})")
                    st.image(tmdb_poster_url(row.get("poster_path")), use_container_width=False)
                    st.write(str(row.get("overview","") or ""))

# ============================= BROWSE ==============================
with tabs[1]:
    st.title("📚 Browse the catalog")

    vocab = set()
    gcol = "genres_text" if "genres_text" in movies.columns else "genres"
    for x in movies[gcol].dropna():
        for g in parse_genres_cell(x):
            if g: vocab.add(g)
    genre_choices = sorted(vocab)

    c1, c2, c3, c4 = st.columns([1,1,2,1])
    with c1:
        years = movies["year"].dropna().astype(int)
        if not years.empty:
            yr_min, yr_max = int(years.min()), int(years.max())
            year_range = st.slider("Year range", yr_min, yr_max, (max(yr_min, 1980), yr_max))
        else:
            year_range = None
    with c2:
        sel_genres = st.multiselect("Genres", genre_choices, default=[])
    with c3:
        q = st.text_input("Search title", "")
    with c4:
        sort_by = st.selectbox("Sort by", ["popularity","year","title"])

    view = movies.copy()
    if year_range:
        view = view[view["year"].between(year_range[0], year_range[1], inclusive="both")]
    if sel_genres:
        def has_all(cell):
            gset = set(map(str.lower, parse_genres_cell(cell)))
            return all(g.lower() in gset for g in sel_genres)
        view = view[view[gcol].apply(has_all)]
    if q.strip():
        view = view[view["title"].str.contains(q.strip(), case=False, na=False)]
    ascending = True if sort_by in ("title","year") else False
    view = view.sort_values(sort_by, ascending=ascending, na_position="last")

    BROWSE_PAGE_SIZE = 25
    total = len(view)
    total_pages = max(1, (total + BROWSE_PAGE_SIZE - 1) // BROWSE_PAGE_SIZE)
    if "browse_page" not in st.session_state:
        st.session_state.browse_page = 1
    left, mid, right = st.columns([1, 4, 1])
    with right:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("◀", key="prev_page", use_container_width=True) and st.session_state.browse_page > 1:
                st.session_state.browse_page -= 1
        with c2:
            st.markdown(
                f"<p style='text-align:center; font-size:1rem; color:#d0cde1;'>Page {st.session_state.browse_page} of {total_pages}</p>",
                unsafe_allow_html=True
            )
        with c3:
            if st.button("▶", key="next_page", use_container_width=True) and st.session_state.browse_page < total_pages:
                st.session_state.browse_page += 1

    pg = st.session_state.browse_page
    start, end = (pg - 1) * BROWSE_PAGE_SIZE, min(pg * BROWSE_PAGE_SIZE, total)
    st.caption(f"{total:,} movies • showing {start+1:,}–{end:,}")

    subset = view.iloc[start:end]
    cols = st.columns(5)
    for j, (_, row) in enumerate(subset.iterrows()):
        with cols[j % 5]:
            title = str(row.get("title",""))
            year = int(row["year"]) if pd.notna(row.get("year")) else None
            st.markdown(f"**{title}**{' ('+str(year)+')' if year else ''}")
            st.image(tmdb_poster_url(row.get("poster_path")), use_container_width=True)
            ov = str(row.get("overview","") or "")
            st.write(ov[:180] + ("…" if len(ov) > 180 else ""))

# ======================= SIMILAR BY TITLE =========================
with tabs[2]:
    st.title("🎯 Similar movies by title")
    seed = st.text_input("Type a movie title you like (e.g., John Wick)", "")
    if st.button("Find similar", type="primary") and seed.strip():
        match = process.extractOne(seed, titles, scorer=fuzz.WRatio)
        if not match or match[1] < 60:
            st.warning("Couldn’t confidently find that title. Try a different spelling.")
        else:
            best_title = match[0]
            seed_rows = movies[movies["title"] == best_title]
            seed_row = seed_rows.iloc[0]
            seed_idx = int(seed_row.name)
            seed_vec = emb[seed_idx:seed_idx+1]

            st.markdown(f"### Seed: {seed_row['title']} ({int(seed_row['year']) if pd.notna(seed_row['year']) else '—'})")
            st.image(tmdb_poster_url(seed_row.get("poster_path")), use_container_width=False)
            st.write(str(seed_row.get("overview","") or ""))

            idxs, sims = knn_query(index, seed_vec, k=TOPK_CAND)
            if len(idxs) and idxs[0] == seed_idx:
                idxs, sims = idxs[1:], sims[1:]
            pops = movies.loc[idxs, "popularity"].values if "popularity" in movies.columns else None
            fused = fuse_scores(sims, idxs, sent01, pops, W_SIM, W_SENT, W_POP)
            order = mmr(seed_vec[0], emb[idxs], np.arange(len(idxs)), fused, lambda_=MMR_LAMBDA, k=TOPK_SHOW) if USE_MMR else np.argsort(-fused)[:TOPK_SHOW]
            final_idxs = idxs[order]; final_sims = sims[order]
            recs = movies.iloc[final_idxs].copy()

            st.markdown("## Similar recommendations")
            st.caption(summarize_results_llm(recs, llm, sent01[final_idxs] if len(final_idxs)>0 else None))
            for rank, (_, row) in enumerate(recs.iterrows(), start=1):
                st.markdown(f"### {rank}. {row['title']} ({int(row['year']) if pd.notna(row['year']) else '—'})")
                st.image(tmdb_poster_url(row.get("poster_path")), use_container_width=False)
                st.write(str(row.get("overview","") or ""))