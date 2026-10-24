FROM python:3.11-slim

WORKDIR /app
ENV HF_HOME=/app/.cache/huggingface \
    ARTIFACTS_DIR=/app/artifacts

# CPU-only PyTorch keeps the image around 1.5 GB instead of 5+ GB.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY movie_recommendation_system ./movie_recommendation_system
# hnswlib ships no wheel for this platform, so compile it, then remove the compiler.
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && pip install --no-cache-dir . \
    && apt-get purge -y g++ && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Bake the sentence model into the image so the container starts without network access.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY artifacts ./artifacts

RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "movie_recommendation_system.api:app", "--host", "0.0.0.0", "--port", "8000"]
