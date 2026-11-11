FROM python:3.12-slim

# EXTRAS="api" is CPU-light (extractive answers, hashing embeddings).
# Build with --build-arg EXTRAS="api,ml" for neural embeddings / LLM generation.
ARG EXTRAS=api

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RAGFC_DATA_DIR=/app/data
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[${EXTRAS}]"

# The index is not baked into the image: mount ./data (see docker-compose.yml).
RUN useradd --create-home app && mkdir -p /app/data && chown app /app/data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "rag_fact_checking.api:app", "--host", "0.0.0.0", "--port", "8000"]
