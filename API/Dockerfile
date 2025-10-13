# API/Dockerfile

FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy only the backend files first (improves caching)
COPY API/ ./API
COPY RAG/ ./RAG
COPY config.py ./
# Copy the data folder
COPY data/ ./data
# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r ./API/requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Default command to run FastAPI
CMD ["uvicorn", "API.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]


