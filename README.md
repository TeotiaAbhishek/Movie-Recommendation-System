# 🎬 NLP-Powered Movie Recommendation System

An intelligent movie recommendation platform that combines **Natural Language Processing (NLP)**, **semantic search**, and **sentiment-aware ranking** to improve movie discovery.

Unlike traditional recommendation systems that rely only on genres or ratings, this system allows users to search naturally using free-text prompts such as:

> "I want something dark and psychological with a mystery theme"

or retrieve recommendations from an existing movie title:

> "John Wick"

The system uses **SBERT embeddings**, **HNSW Approximate Nearest Neighbour search**, **sentiment analysis**, and **reranking techniques** to deliver relevant recommendations in real time.

---

## 🚀 Features

### 1. Free-text Recommendation Mode
Users can describe movies using natural language.

Example:

```text
Looking for a sci-fi movie with space exploration and emotional storytelling
```

The system interprets intent semantically instead of relying on keywords.

---

### 2. Similar Movies by Title

Users can enter a movie title and retrieve semantically similar recommendations.

Example:

```text
Input: John Wick
Output: John Wick Chapter 2, Nobody, Atomic Blonde
```

Supports:

- Exact matching
- Fuzzy matching
- NLP-based title extraction

---

### 3. Browse Mode

Interactive browsing interface supporting:

- Genre filtering
- Release year filtering
- Popularity sorting
- Catalog exploration

---

### 4. Sentiment-aware Ranking

Recommendations combine:

- Semantic similarity
- Audience sentiment
- Popularity signals

Final score:

```python
Final Score =
0.7 × Semantic Similarity +
0.2 × Sentiment +
0.1 × Popularity
```

---

## 🏗 Architecture

```text
User Input (Title / Free Text)
            ↓
NLP Title Extraction / Embedding
            ↓
SBERT Embeddings
            ↓
HNSW ANN Search
            ↓
Score Fusion
(Similarity + Sentiment + Popularity)
            ↓
MMR Reranking
            ↓
Explanation Layer
            ↓
Streamlit Interface
```

---

## 🛠 Tech Stack

### NLP & Recommendation

- Sentence Transformers (all-MiniLM-L6-v2)
- SBERT Embeddings
- HNSW (hnswlib)
- VADER Sentiment Analysis
- spaCy
- RapidFuzz
- Regex

### Backend

- Python
- NumPy
- Pandas
- FastAPI (planned)

### Frontend

- Streamlit

### Models & Retrieval

- Semantic Search
- Approximate Nearest Neighbour Retrieval
- Maximal Marginal Relevance (MMR)
- Score Fusion

---

## 📊 Dataset

Source:

- TMDb
- IMDb metadata
- Kaggle movie datasets

Dataset includes:

- Movie titles
- Overview text
- Genres
- Popularity metrics
- Ratings
- Reviews

Scale:

- **44,471 movies**
- **384-dimensional embeddings**
- **Sub-second retrieval latency**

---

## 📁 Project Structure

```text
movie_recommendation_system/

│── app/
│   ├── home.py
│   ├── browse.py
│   └── recommendation_ui.py
│
│── models/
│   ├── embedding_model.py
│   ├── retrieval.py
│   ├── sentiment.py
│   └── reranking.py
│
│── api/
│   ├── extract_api.py
│   ├── recommendation_api.py
│   └── sentiment_api.py
│
│── artifacts/
│   ├── embeddings.npy
│   ├── hnsw_cosine.idx
│   ├── review_sentiment_01.npy
│   ├── movies_aug.parquet
│   └── id_map.json
│
│── data/
│
│── requirements.txt
│── README.md
```

---

## ⚡ Performance

| Metric | Result |
|----------|---------|
| Movies processed | 44,471 |
| Embedding model | SBERT all-MiniLM-L6-v2 |
| Embedding size | 384 |
| Retrieval method | HNSW ANN |
| Query latency | < 0.5 sec |
| Search type | Semantic |

---

## 📈 Key Outcomes

✅ Real-time semantic recommendation engine

✅ Sub-second retrieval over 44k+ movies

✅ Multi-modal search:

- Free text
- Title-based
- Browse mode

✅ Sentiment-enhanced recommendation quality

✅ Scalable ANN retrieval pipeline

---

## 🔮 Future Improvements

Planned enhancements:

- Collaborative filtering
- User profile personalization
- Cloud deployment
- FastAPI production APIs
- Advanced sentiment models (RoBERTa / DistilBERT)
- Hybrid recommendation engine
- Online inference pipeline

---

## 👥 Team

### Recommendation Model & Testing
**Abhishek Teotia**

- Recommendation engine
- Retrieval pipeline
- API integration
- System testing

### NLP Query Extraction
Muskan Choudhary

### Sentiment Module
Justin Govanxa

### Frontend & LLM Integration
Yuzhong Wang

---

## 📷 Application Preview

### Free-text Search

```text
"I want a horror movie with supernatural elements"
```

### Similar Movies

```text
Input: John Wick
```

### Browse Mode

Genre + Popularity + Release Year filtering

---

## 📜 License

Educational and research use only.

Built as part of:

**36118 Applied Natural Language Processing**  
University of Technology Sydney
