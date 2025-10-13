# UI/Dockerfile

FROM python:3.12-slim

WORKDIR /app

# Copy frontend files
COPY UI/ ./UI
COPY RAG/ ./RAG
COPY config.py ./

# Upgrade pip
RUN pip install --upgrade pip

# Install frontend dependencies
RUN pip install -r ./UI/requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Default command to run Streamlit
CMD ["streamlit", "run", "UI/streamlit_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]



