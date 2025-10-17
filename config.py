import os

# 🧠 Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


# 📄 Files
AXRIV_PDF=os.path.join(DATA_DIR, "axriv_papers")
METADATA_JSON = os.path.join(DATA_DIR, "metadata.json")
EXTRACTED_JSON = os.path.join(DATA_DIR, "extracted_content.json")
CHUNKED_JSON = os.path.join(DATA_DIR, "chunked_dataset_scibert.json")
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")

# 🧭 Models
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
GENERATOR_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# ⚙️ RAG parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 10
MAX_TOKENS = 256

# 🌐 API settings
FASTAPI_HOST = "0.0.0.0"
FASTAPI_PORT = 8000


#Evaluation Parameters

DEFAULT_METRICS = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]

# Paths
RESULTS_DIR = "evaluation/results"
DEFAULT_CSV_PATH = f"{RESULTS_DIR}/results.csv"

# LangSmith project (optional)
LANGSMITH_PROJECT = "rag-eval-experiments"
