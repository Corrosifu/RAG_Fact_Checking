from typing import List
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import numpy as np
from transformers import AutoTokenizer
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders.huggingface import HuggingFaceCrossEncoder
from langchain.docstore.document import Document
from Chunking import SciBERTChunker


class Retrieval:
    """
    Hybrid retrieval class combining dense (FAISS) and sparse (BM25) retrieval,
    with optional cross-encoder reranking. Handles device placement to avoid
    CPU/GPU tensor mismatches.
    """

    def __init__(self, faiss_index_path: str, sparse_corpus: List[str], device: str = None):
        self.faiss_index_path = faiss_index_path
        self.sparse_corpus = sparse_corpus
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_models_and_indexes()

    def load_models_and_indexes(self):
        # Dense embeddings (HuggingFace) always use CPU for FAISS
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 1},
        )

        # Load FAISS index (CPU)
        self.faiss_db = FAISS.load_local(
            self.faiss_index_path,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True,
        )
        self.dense_retriever = self.faiss_db.as_retriever(search_kwargs={"k": 5})

        # Sparse BM25
        self.scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        tokenized_corpus = [self.scibert_tokenizer.tokenize(doc.lower()) for doc in self.sparse_corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Cross-encoder reranker on CPU to match embeddings
        cross_encoder_model = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            model_kwargs={"device": "cpu"}
        )
        self.cross_encoder_reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=10)

    # --------------------
    # Sparse retrieval
    # --------------------
    def sparse_query(self, query: str) -> List[str]:
        return self.scibert_tokenizer.tokenize(query.lower())

    def retrieve_sparse(self, sparse_query_tokens: List[str]) -> List[Document]:
        scores = self.bm25.get_scores(sparse_query_tokens)
        top_n = np.argsort(scores)[::-1][:10]
        return [Document(page_content=self.sparse_corpus[i], metadata={}) for i in top_n]

    # --------------------
    # Dense retrieval
    # --------------------
    def retrieve_dense(self, query: str) -> List[Document]:
        results = self.dense_retriever.invoke(query)  # embeddings handled internally on CPU
        return [
            Document(page_content=r.page_content, metadata=r.metadata if hasattr(r, "metadata") else {})
            for r in results
        ]

    # --------------------
    # Merge & rerank
    # --------------------
    def merge_results(self, results_dense: List[Document], results_sparse: List[Document]) -> List[Document]:
        seen = set()
        merged = []
        for doc in results_dense + results_sparse:
            if doc.page_content not in seen:
                merged.append(doc)
                seen.add(doc.page_content)
        return merged

    def rerank(self, merged_results: List[Document], query: str) -> List[Document]:
        return self.cross_encoder_reranker.compress_documents(merged_results, query)

    # --------------------
    # Full pipeline
    # --------------------
    def retrieve(self, query: str) -> List[Document]:
        sparse_tokens = self.sparse_query(query)
        dense_docs = self.retrieve_dense(query)
        sparse_docs = self.retrieve_sparse(sparse_tokens)
        merged = self.merge_results(dense_docs, sparse_docs)
        return self.rerank(merged, query)


    def run():
        chunked_dataset_path = "arxiv_papers/chunked_dataset_scibert.json"
        chunked_dataset = SciBERTChunker.load_articles(chunked_dataset_path)
        corpus_sparse = [chunk["text"] for chunk in chunked_dataset]
        return Retrieval(faiss_index_path="faiss_index", sparse_corpus=corpus_sparse)
