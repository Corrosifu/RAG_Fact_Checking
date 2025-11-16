<div align="center">

# 🧠 RAG Fact‑Checking System

[![Made with Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-109989?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-1B6AC6.svg)](https://faiss.ai/)
[![HuggingFace](https://img.shields.io/badge/Models-HuggingFace-F7C713?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Transformers](https://img.shields.io/badge/NLP-Transformers-2C2D72?logo=huggingface&logoColor=white)](https://github.com/huggingface/transformers)
[![arXiv](https://img.shields.io/badge/Data-arXiv-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/)
[![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Azure](https://img.shields.io/badge/Cloud-Azure-0078D4?logo=microsoft-azure&logoColor=white)](https://azure.microsoft.com/)
[![Project Status](https://img.shields.io/badge/Status-Alpha-orange)](#)
[![Contributions welcome](https://img.shields.io/badge/Contributions-welcome-brightgreen.svg)](#)

**Author:** [Corrosifu](https://github.com/Corrosifu) • **Goal:** Build a *lightweight*, *explainable*, and *modular* Retrieval‑Augmented Generation (RAG) system for scientific fact‑checking and knowledge explanation — fully executable on limited hardware (≈ **4 GB VRAM + CPU**).

</div>

---

## 🔎 Table of Contents

- [Context & Business Problem](#-1-context--business-problem)
- [Project Goals](#-2-project-goals)
- [Architecture Overview](#-3-architecture-overview)
  - [Modules Breakdown](#-31-modules-breakdown)
- [Technical Choices & Justifications](#-4-technical-choices--justifications)
- [Execution Pipeline](#-5-execution-pipeline)
- [Example Usage](#-6-example-usage)
- [Limitations](#-7-limitations)
- [Possible Improvements](#-8-possible-improvements)
- [Environment & Setup](#-9-environment--setup)
  - [Docker](#2-launch-the-app-on-docker)
  - [Azure](#4-launch-the-app-on-azure)
- [Why This Project Matters](#-10-why-this-project-matters)

---

## 📍 1. Context & Business Problem

The explosive growth of scientific publications has made it almost impossible to manually validate claims or summarize relevant research.  
This project aims to automate **scientific knowledge retrieval and explanation** using modern AI techniques — without depending on commercial APIs or large-scale cloud resources.

**Business Use‑Case:**  
- Journalists and researchers often need **to verify or explain a scientific claim** quickly.  
- The system allows them to query topics (e.g., *"How does quantization work?"*) and receive contextual, sourced, and simplified explanations built from **real arXiv papers**.

**Strategic Objective:**  
Provide a reproducible, open‑source RAG pipeline that:
- Automates scientific paper ingestion and structuring.
- Creates searchable vector representations of knowledge.
- Generates understandable explanations from retrieved documents.
- Runs efficiently on modest local hardware.

> **Hardware context:** Implemented and tested on **~4 GB VRAM + CPU**, prioritizing compact models and memory‑safe defaults.

---

## 🎯 2. Project Goals

1. **Automate ingestion** of scientific papers from [arXiv](https://arxiv.org).  
2. **Extract and structure** the content into Markdown and JSON datasets.  
3. **Chunk and embed** text for efficient retrieval using *SciBERT* and *Qwen embeddings*.  
4. **Index with FAISS** for semantic similarity search.  
5. **Combine dense and sparse retrieval** (FAISS + BM25) for robustness.  
6. **Generate pedagogical explanations** with a small LLaMA‑based model.  
7. **Ensure low resource footprint** for CPU/GPU‑constrained environments.

---

## ⚙️ 3. Architecture Overview

```
arXiv → PDF ingestion → Markdown extraction → SciBERT chunking
      → FAISS + BM25 hybrid retrieval → Cross‑encoder reranking
      → LLaMA generator → Pedagogical explanation
```

<p align="center">
  <img src="img/diagram"
       alt="RAG Fact-Checking Architecture"
       width="900">
  <br>
  <em>Figure — End-to-end RAG pipeline from arXiv ingestion to retrieval and generation.</em>
</p>



### 📦 Repository Structure

```text
📁 RAG_FACT_CHECKING/
│
├─ API/                     # Backend
│  ├─ fastapi_app.py
│  ├─ requirements.txt
│  └─ Dockerfile
├─ UI/                      # Frontend
│  ├─ streamlit_ui.py
│  ├─ requirements.txt
│  └─ Dockerfile
├─ RAG/                     # RAG process from data collect to generation phase
│  ├─ ingestion.py
│  ├─ chunking.py
│  ├─ embedding.py
│  ├─ data_pipeline.py
│  ├─ retriever.py
│  ├─ generator.py
│  └─ rag.py
├─ Evaluation/              # Evaluate RAG performances
│  ├─ dataset_builder.py
│  ├─ logger.py
│  ├─ metric_runner.py
│  ├─ llm_judge.py
│  ├─ vizualization.py
│  ├─ evaluation.py
│  └─ test.ipynb
│
├─ config.py
├─ docker-compose.yml       # On‑prem deployment
└─ docker-compose.prod.yml  # Azure deployment
```

### 🔹 3.1 Modules Breakdown

| Module | Purpose | Key Libraries |
|--------|--------|---------------|
| `ArxivIngestor` | Fetches and parses scientific papers | `feedparser`, `requests`, `pdfplumber`, `pymupdf4llm` |
| `SciBERTChunker` | Token‑level text segmentation | `transformers`, `allenai/scibert_scivocab_uncased` |
| `Embedder` | Converts text chunks to dense embeddings | `FAISS`, `langchain`, `Qwen/Qwen3-Embedding-0.6B` |
| `Retrieval` | Combines FAISS + BM25 + Cross‑Encoder reranking | `rank_bm25`, `langchain`, `MiniLM` |
| `Generator` | Generates concise explanations | `meta-llama/Llama‑3.2‑1B‑Instruct` |
| `DataPipeline` | Automates ingestion → chunking → embedding | Orchestrator |
| `RAG` | End‑user interface for QA | Integrates retriever + generator |

---

## 🧩 4. Technical Choices & Justifications

### 🧠 **1. SciBERT for Domain Tokenization**
- Tailored for scientific text; improves term segmentation (`gradient descent`, `transformer`, etc.).
- Reduces semantic drift during embedding and BM25 matching.

### 🪶 **2. Qwen 3 Embedding Model (0.6 B)**
- Compact, multilingual model with normalized embeddings.
- Strong retrieval performance while fitting into **4 GB VRAM**.
- Great trade‑off for lightweight local inference.

### 📚 **3. FAISS + BM25 Hybrid Retrieval**
- **Dense retrieval** (FAISS) captures semantic meaning.  
- **Sparse retrieval** (BM25) ensures lexical precision.  
- **Cross‑encoder reranking** (MiniLM) reorders top hits for precision — balancing recall & accuracy.

### 🧾 **4. LLaMA 3.2‑1B‑Instruct as Generator**
- Instruction‑tuned for *teaching‑style* answers.
- Runs on CPU or small GPUs using FP16/FP32.
- Custom prompt encourages clarity and avoids verbatim copying.

### 🧱 **5. Modular Architecture**
- Clear interfaces between ingestion, chunking, embedding, retrieval, and generation.
- Easy to swap models, indexes, or evaluation components.

### 🔬 **6. Evaluation Pipeline**
- **Coming soon** (module stubs present under `/Evaluation`).  
  Targets: retrieval metrics (MRR / Recall@k), generation quality (LLM‑as‑judge), dashboards.

---

## 🧮 5. Execution Pipeline

```bash
# Full data pipeline (ingestion → chunking → embedding)
python -m RAG.data_pipeline

# Run the RAG (retriever + generator)
python -m RAG.rag
```


---

## 🧠 6. Example Usage

```python
from RAG.rag import RAG

rag = RAG()
response = rag.ask("How does quantization work in deep learning?")
print(response)
```

**Example Output:**
> Quantization reduces model size by representing weights and activations with fewer bits.  
> For instance, 32‑bit floats can be approximated with 8‑bit integers, saving memory and computation.  
> This trade‑off slightly affects accuracy but is essential for edge deployment.

<p align="center">
  <img src="img/streamlitui.png"
       alt="UI"
       width="900">
  <br>
  <em>Figure — Example of an output </em>
</p>
---

## 📉 7. Limitations

| Area                       | Limitation                                             | Explanation                                                                                      |
| -------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| **Hardware Constraints**   | Works with ~4 GB VRAM                                  | Designed for lightweight models; larger ones (e.g., 7B+) exceed memory capacity.                 |
| **Model Scope**            | Processes only English arXiv papers (no images/tables) | Current ingestion pipeline supports text-only data and lacks multilingual or multimodal support. |
| **Retrieval Performance**  | BM25 and embedding computation run on CPU              | No GPU acceleration or model quantization, which limits retrieval speed.                         |
| **Generator Context Size** | Context limited to ~1,500 characters                   | Restriction prevents out-of-memory (OOM) errors on smaller GPUs.                                 |
| **Fact Verification**      | Evaluation module not fully integrated                 | Automated verification and scoring of factual consistency are planned but not yet implemented.   |
| **Data Persistence**       | Results stored locally via FAISS                       | No cloud database or persistent UI yet; data resets between sessions.                            |

---

## 🚀 8. Possible Improvements

1. **Citation verification**: align generated claims with arXiv metadata and passages.  
2. **LoRA fine‑tuning** to improve explanation clarity and style.  
3. **Parallelized embedding** (multiprocessing) for CPU efficiency.  
4. **GPU FAISS** or **Milvus** for production‑scale search.  
5. **Metadata filters** (year, field, author) in retriever.  
6. **Quantized models** for both embeddings and generator (e.g., `gguf`, `bitsandbytes`).  
7. **Evaluation suite** with dashboards (Streamlit) and reproducible seeds.

---

## 🧰 9. Environment & Setup

### 1) Run locally (API + UI)

```bash
git clone https://github.com/Corrosifu/RAG_Fact_Checking.git
cd RAG_Fact_Checking

# (Optional) set your HF token for gated models
export HF_TOKEN="hf_xxx_your_token_here"   # Linux/Mac
# setx HF_TOKEN "hf_xxx_your_token_here"   # Windows (PowerShell: $env:HF_TOKEN="...")

# Start backend & UI
python -m uvicorn API.fastapi_app:app --reload --host 127.0.0.1 --port 8000
streamlit run UI/streamlit_ui.py
```

### 2) Launch the app on Docker

```bash
docker-compose build
docker-compose up
```

### 3) Configuration

Edit `config.py` to adjust paths and filenames, for example:
```python
AXRIV_PDF      = "data/pdfs"                 # directory for downloaded PDFs
DATA_DIR       = "data"
EXTRACTED_JSON = "data/extracted_content.json"
METADATA_JSON  = "data/metadata.json"
INDEX_DIR      = "data/faiss_index"
CHUNKED_JSON   = "data/chunked_dataset_scibert.json"
```
> **Note:** the constant name `AXRIV_PDF` is intentionally preserved to match the codebase.

### 4) Launch the app on Azure

After login:

```powershell
$env:HF_TOKEN = "hf_xxx_your_token_here"

$RG  = "rg-rag-portfolio"
$LOC = "francecentral"
$APP = "rag-portfolio-app"
$ACR = "ragportfolioacr"   # pick a unique name

az group create -n $RG -l $LOC

az acr create -n $ACR -g $RG --sku Standard
$REG_LOGIN_SERVER = az acr show -n $ACR -g $RG --query loginServer -o tsv

az acr build -r $ACR -t "$REG_LOGIN_SERVER/rag-backend:latest"  -f API/Dockerfile .
az acr build -r $ACR -t "$REG_LOGIN_SERVER/rag-frontend:latest" -f UI/Dockerfile .

az appservice plan create -g $RG -n "asp-rag" --is-linux --sku P1v3
az webapp create -g $RG -p "asp-rag" -n $APP `
  --multicontainer-config-type compose `
  --multicontainer-config-file docker-compose.prod.yml

az webapp identity assign -g $RG -n $APP
$PRINCIPAL_ID = az webapp identity show -g $RG -n $APP --query principalId -o tsv
$ACR_ID       = az acr show -n $ACR -g $RG --query id -o tsv
az role assignment create --assignee $PRINCIPAL_ID --role "AcrPull" --scope $ACR_ID

$STG = "stragcache$([System.Guid]::NewGuid().ToString('N').Substring(0,8))"
az storage account create -g $RG -n $STG -l $LOC --sku Standard_LRS
az storage share-rm create -g $RG --storage-account $STG --name hf-cache
$STG_KEY = az storage account keys list -g $RG -n $STG --query "[0].value" -o tsv

az webapp config storage-account add -g $RG -n $APP `
  --custom-id hf-cache --storage-type AzureFiles `
  --account-name $STG --share-name hf-cache --access-key "$STG_KEY" `
  --mount-path /hf_cache

az webapp config appsettings set -g $RG -n $APP --settings `
  REG_LOGIN_SERVER=$REG_LOGIN_SERVER `
  HF_TOKEN="$env:HF_TOKEN"

az webapp config container set -g $RG -n $APP `
  --multicontainer-config-type compose `
  --multicontainer-config-file docker-compose.prod.yml
```

---

## 💡 10. Why This Project Matters

This project demonstrates **applied AI architecture design under real constraints**.  
It showcases:
- Mastery of **NLP pipeline engineering** (ingestion → retrieval → generation).  
- Understanding of **retrieval models and embeddings**.  
- The ability to make *smart trade‑offs* between model accuracy and resource usage.  
- Strong focus on **explainability** and **pedagogical clarity**.

It’s an ideal project for demonstrating **full‑stack ML and NLP engineering** competence.

---

<div align="center">

**If this project helps you, consider giving it a ⭐ and sharing feedback via Issues!**

</div>
