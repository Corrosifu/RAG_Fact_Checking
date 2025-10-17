# 🧠 Scientific QA — RAG Evaluation Pipeline

## 📌 Description

The goal of this project is to design, implement, and evaluate a **Retrieval-Augmented Generation (RAG)** system specialized in **scientific question answering**.  
Unlike classic LLM-based QA, the RAG approach combines **external knowledge retrieval** with **generative reasoning**, improving **faithfulness** and **relevance** — key in scientific and technical domains.

This project also provides a **robust evaluation framework** using RAGAS metrics, making it possible to **quantify performance** and adapt models to various business or research contexts.

---

## 📑 Table of Contents

- [Description](#-description)
- [1. Test Structuring and Use Cases](#1-test-structuring-and-use-cases)
- [2. Datasets and Metric Choices](#2-datasets-and-metric-choices)
- [3. Data and Document Preprocessing](#3-data-and-document-preprocessing)
- [4. Method Evaluation](#4-method-evaluation)
  - [Retriever Strategy](#retriever-strategy)
  - [Generator Strategy](#generator-strategy)
  - [RAG Evaluation with RAGAS](#rag-evaluation-with-ragas)
- [5. Constraints and Future Directions](#5-constraints-and-future-directions)
- [Installation and Prerequisites](#installation-and-prerequisites)
- [Usage](#usage)
- [Authors and Contributions](#authors-and-contributions)
- [Resources and References](#resources-and-references)

---

## 🧪 1. Test Structuring and Use Cases

📁RAG_FACT_CHECKING/
│
├─ API/# backend
│   ├─ fastapi_app.py
│   └─ requirements.txt
|   └─ dockerfile
├─ UI/# frontend
│   ├─ streamlit_ui.py
│   └─ requirements.txt
|   └─ dockerfile
├─ RAG/# RAG process from data collect to generation phase
│   ├─ ingestion.py
|   └─ chunking.py
|   └─ embedding.py
|   └─ data_pipeline.py
|   └─ retriever.py
|   └─ generator.py
│   └─ rag.py
├─ Evaluation/ #Evaluate the rag performances 
│   ├─ dataset_builder.py
|   └─ logger.py
|   └─ metric_runner.py
|   └─ llm_judge.py
|   └─ vizualization.py
|   └─ evaluation.py
│   └─ test.ipynb
|   
└─ config.py
└─ docker-compose.yml # On prem deployment 
└─ docker-compose.prod.yml # Azure deployment

- Development of a **scientific question-answering system** combining:
  - Semantic document retrieval
  - Lightweight or open-source language models
  - Grounded answer generation
- Modular architecture allowing:
  - Easy swapping of retrievers and LLMs
  - Batch evaluation of test queries
- Typical use cases:
  - 🏢 **Companies** building internal scientific assistants
  - 🧠 **Researchers** evaluating factual grounding
  - 🧪 **Developers** experimenting with RAG pipelines on limited hardware (4GB VRAM)

---

## 📊 2. Datasets and Metric Choices

- Integration of domain-specific scientific texts or public knowledge corpora.
- Vectorization using dense embeddings (e.g. `sentence-transformers`).
- Sample test set: handcrafted QA pairs for evaluation.
- Evaluation metrics via **RAGAS**:
  - **Faithfulness** — factual alignment with retrieved documents
  - **Context precision & recall** — retrieval quality
  - **Answer relevancy** — user-centric response evaluation

---

## 🧹 3. Data and Document Preprocessing

- Document cleaning and normalization (HTML removal, lowercasing, sentence segmentation).
- Embedding generation via Hugging Face models (e.g. MiniLM, BGE-small).
- FAISS index construction for efficient similarity search.
- Optional metadata filtering to simulate realistic scientific search.

---

## 🧠 4. Method Evaluation

### 🧭 Retriever Strategy

- Implementation of a **dense retriever** (FAISS) to index and search relevant documents.
- Comparison with potential BM25 or hybrid strategies.
- Focus on:
  - Retrieval accuracy
  - Context coverage
  - Efficiency on consumer hardware

### 🪄 Generator Strategy

- Lightweight **open-source LLMs** (e.g. TinyLlama, DistilGPT2) used for local inference.
- Option to plug larger or cloud-hosted models for benchmarking.
- Response generation designed to stay aligned with retrieved context.

### 📈 RAG Evaluation with RAGAS

| Metric                | Description                                          | Insight                                  |
|-----------------------|------------------------------------------------------|-------------------------------------------|
| Faithfulness          | Measures factual grounding of the answer             | Detects hallucinations                    |
| Context Precision     | Relevance of retrieved context                       | Retrieval quality                         |
| Context Recall        | Coverage of relevant context                         | Completeness of context                   |
| Answer Relevancy      | Alignment of answer with expected reference          | User satisfaction                         |

- Evaluation is automated through the `Eval_Pipeline` class.
- Results stored as a Pandas DataFrame and visualized through custom plots.

Example output:

| user_input                     | faithfulness | context_precision | context_recall | answer_relevancy | mean_score |
|----------------------------------|--------------|--------------------|----------------|------------------|------------|
| What is RAG in ML?             | 0.89         | 0.92               | 0.87           | 0.90             | 0.89       |

---

## ⚠️ 5. Constraints and Future Directions

- **Resource constraints**: optimized for 4GB VRAM setups, limiting LLM and database size but also embedding and retrieving strategy.
- Open-source models offer flexibility but lower raw performance vs GPT-4.
- RAGAS metrics rely on LLM grading — subject to rate limits or cost for API-based models.
- Future improvements:
  - Support for structured multimodal data (tables, graphs, images)
  - Multi sources scientific papers
  - Multi fields scientific papers
  - Fine-tuning of domain-specific embedding models
  - Distributed evaluation for larger QA sets
  - Integration of guardrails and fact-checking modules
    
  
---

## ⚙️ Installation and Prerequisites


1. **Clone the repository and launch the app locally**
```bash
git clone https://github.com/Corrosifu/RAG_Fact_Checking.git
cd RAG_Fact_Checking
python -m uvicorn API.fastapi_app:app --reload --host 127.0.0.1 --port 8000
python -m streamlit run UI/streamlit_ui.py
```
2. **Launch the app on docker**
```bash
docker-compose build
docker-compose up
```

4. **Launch the app on Azure**
---
After login
```bash
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
az webapp restart -g $RG -n $APP


az webapp show -g $RG -n $APP --query defaultHostName -o tsv
```
