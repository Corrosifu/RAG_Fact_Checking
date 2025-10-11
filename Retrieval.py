from typing import List
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import numpy as np
from transformers import AutoTokenizer
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders.huggingface import HuggingFaceCrossEncoder
from Chunking import load_articles
import pandas as pd
import matplotlib.pyplot as plt
from langchain.docstore.document import Document
class Retrieval:
    def __init__(self, faiss_index_path: str, sparse_corpus: List[str]):
        self.faiss_index_path = faiss_index_path
        self.sparse_corpus = sparse_corpus
        self.load_models_and_indexes()

    def load_models_and_indexes(self): 
        device={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            model_kwargs=device,
            encode_kwargs={"normalize_embeddings": True,"batch_size": 1}
        )
        self.faiss_db = FAISS.load_local(self.faiss_index_path, embeddings=self.embedding_model, allow_dangerous_deserialization=True)
        self.dense_retriever = self.faiss_db.as_retriever(search_kwargs={"k": 10})
        self.scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        tokenized_corpus = [self.scibert_tokenizer.tokenize(doc.lower()) for doc in self.sparse_corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        cross_encoder_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", model_kwargs=device)
        self.cross_encoder_reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=10)

    def embed_query(self, query: str):
        return query  #self.embedding_model.embed_query(query)

    def sparse_query(self, query: str):
        return self.scibert_tokenizer.tokenize(query.lower())

    def retrieve_dense(self, embedded_query: np.ndarray):
        return self.dense_retriever.invoke(embedded_query)

    def retrieve_sparse(self, sparse_query_tokens: List[str]):
        scores = self.bm25.get_scores(sparse_query_tokens)
        top_n = np.argsort(scores)[::-1][:10]
        return [self.sparse_corpus[i] for i in top_n]

    def merge_results(self, results_dense: List[str], results_sparse: List[str]) -> List[str]:
        seen = set()
        merged = []
        results_sparse=[Document(page_content=text,metadata={}) for text in results_sparse]
        for doc in results_dense + results_sparse:
            identifier=doc.page_content

            if identifier not in seen:
                merged.append(doc)
                seen.add(identifier)
        return merged

    def rerank(self, merged_results: List[str], query: str) -> List[str]:
        return self.cross_encoder_reranker.compress_documents(merged_results, query)

    def retrieve(self, query: str) -> List[str]:
        embedded_query = self.embed_query(query)
        sparse_query_tokens = self.sparse_query(query)
        results_dense = self.retrieve_dense(embedded_query)
        results_sparse = self.retrieve_sparse(sparse_query_tokens)
        merged = self.merge_results(results_dense, results_sparse)
        return self.rerank(merged, query)




def main():

    chunked_dataset_path = "arxiv_papers/chunked_dataset_scibert.json"
    chunked_dataset = load_articles(chunked_dataset_path)
    corpus_sparse = [chunk["text"] for chunk in chunked_dataset]
    return  Retrieval(faiss_index_path="faiss_index", sparse_corpus=corpus_sparse)
    



if __name__ == "__main__":

    query='How do RAG works'
    print(main().retrieve(query))
