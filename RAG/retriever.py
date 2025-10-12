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
from RAG.chunking import SciBERTChunker
from config import INDEX_DIR, CHUNKED_JSON
class Retrieval:
    """
    A hybrid document retrieval pipeline combining:
      - Dense retrieval (via FAISS + HuggingFace embeddings)
      - Sparse retrieval (via BM25 using SciBERT tokenization)
      - Optional cross-encoder reranking for higher precision.

    This class is designed for scientific or research-based corpora,
    leveraging SciBERT for tokenization and embeddings tuned for
    scientific text understanding.

    Attributes
    ----------
    faiss_index_path : str
        Path to a pre-built FAISS index for dense vector search.
    sparse_corpus : List[str]
        A list of documents or text chunks used for BM25 retrieval.
    device : str
        Device on which computation is performed ('cpu' or 'cuda').
    embedding_model : HuggingFaceEmbeddings
        Dense embedding model used for FAISS retrieval.
    faiss_db : FAISS
        Loaded FAISS vector store for dense retrieval.
    dense_retriever : langchain retriever
        Dense retriever interface for top-k FAISS search.
    scibert_tokenizer : AutoTokenizer
        Tokenizer for SciBERT model used in BM25 preprocessing.
    bm25 : BM25Okapi
        Sparse retriever scoring model.
    cross_encoder_reranker : CrossEncoderReranker
        Reranker model to refine retrieved results.

    Methods
    -------
    load_models_and_indexes():
        Loads dense/sparse retrievers and cross-encoder reranker.
    sparse_query(query: str) -> List[str]:
        Tokenizes a query for BM25 retrieval.
    retrieve_sparse(sparse_query_tokens: List[str]) -> List[Document]:
        Retrieves top-N sparse results from the BM25 index.
    retrieve_dense(query: str) -> List[Document]:
        Retrieves top-N dense results from the FAISS vector store.
    merge_results(results_dense, results_sparse) -> List[Document]:
        Merges results from both retrievers, removing duplicates.
    rerank(merged_results, query) -> List[Document]:
        Reranks combined results using a cross-encoder.
    retrieve(query: str) -> List[Document]:
        Executes the full retrieval pipeline: dense + sparse + rerank.
    run() -> 'Retrieval':
        Factory method to instantiate the retriever using a saved corpus.
    """

    def __init__(self, chunked_dataset=SciBERTChunker.load_articles(CHUNKED_JSON), device: str = None):
        """
        Initialize the retrieval pipeline.

        Parameters
        ----------
        chunked_dataset : List[dict]
            List of Document.
        device : str, optional
            Device type ('cpu' or 'cuda'), auto-detected if not provided.
        """
        
        self.chunked_dataset = chunked_dataset
        self.corpus_sparse = [chunk["text"] for chunk in chunked_dataset]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_models_and_indexes()

    def load_models_and_indexes(self):
        """Load all required models (dense, sparse, and reranker) and indexes."""
        # Load dense embedding model for FAISS — always CPU to avoid FAISS CUDA issues
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 1},
        )

        # Load FAISS vector store (dense retrieval)
        self.faiss_db = FAISS.load_local(
            INDEX_DIR,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True,
        )
        self.dense_retriever = self.faiss_db.as_retriever(search_kwargs={"k": 5})

        # Initialize BM25 sparse retrieval with SciBERT tokenizer
        self.scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        tokenized_corpus = [self.scibert_tokenizer.tokenize(doc.lower()) for doc in self.corpus_sparse]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Load cross-encoder reranker (kept on CPU to match embedding device)
        cross_encoder_model = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            model_kwargs={"device": "cpu"}
        )
        self.cross_encoder_reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=10)

    # --------------------
    # Sparse retrieval
    # --------------------
    def sparse_query(self, query: str) -> List[str]:
        """Tokenize a query for BM25 retrieval."""
        return self.scibert_tokenizer.tokenize(query.lower())

    def retrieve_sparse(self, sparse_query_tokens: List[str]) -> List[Document]:
        """
        Retrieve documents using BM25 based on sparse lexical matching.

        Parameters
        ----------
        sparse_query_tokens : List[str]
            Tokenized query.

        Returns
        -------
        List[Document]
            Top 10 BM25-ranked documents.
        """
        scores = self.bm25.get_scores(sparse_query_tokens)
        top_n = np.argsort(scores)[::-1][:10]
        return [Document(page_content=self.corpus_sparse[i], metadata={}) for i in top_n]

    # --------------------
    # Dense retrieval
    # --------------------
    def retrieve_dense(self, query: str) -> List[Document]:
        """
        Retrieve documents from FAISS dense index based on vector similarity.

        Parameters
        ----------
        query : str
            User query.

        Returns
        -------
        List[Document]
            Top-k dense retrieval results.
        """
        results = self.dense_retriever.invoke(query)  # FAISS handles embedding on CPU
        return [
            Document(page_content=r.page_content, metadata=getattr(r, "metadata", {}))
            for r in results
        ]

    # --------------------
    # Merge & rerank
    # --------------------
    def merge_results(self, results_dense: List[Document], results_sparse: List[Document]) -> List[Document]:
        """
        Merge dense and sparse retrieval results while removing duplicates.

        Parameters
        ----------
        results_dense : List[Document]
            Dense retrieval output.
        results_sparse : List[Document]
            Sparse retrieval output.

        Returns
        -------
        List[Document]
            Combined unique set of documents.
        """
        seen = set()
        merged = []
        for doc in results_dense + results_sparse:
            if doc.page_content not in seen:
                merged.append(doc)
                seen.add(doc.page_content)
        return merged

    def rerank(self, merged_results: List[Document], query: str) -> List[Document]:
        """
        Rerank merged results using a cross-encoder model for semantic relevance.

        Parameters
        ----------
        merged_results : List[Document]
            Combined list of documents.
        query : str
            Query text.

        Returns
        -------
        List[Document]
            Reranked top-n results.
        """
        return self.cross_encoder_reranker.compress_documents(merged_results, query)

    # --------------------
    # Full pipeline
    # --------------------
    def retrieve(self, query: str) -> List[Document]:
        """
        Perform the full hybrid retrieval pipeline.

        Combines:
          1. Sparse BM25 lexical retrieval
          2. Dense FAISS semantic retrieval
          3. Cross-encoder reranking for precision

        Parameters
        ----------
        query : str
            User query.

        Returns
        -------
        List[Document]
            Final reranked list of relevant documents.
        """
        sparse_tokens = self.sparse_query(query)
        dense_docs = self.retrieve_dense(query)
        sparse_docs = self.retrieve_sparse(sparse_tokens)
        merged = self.merge_results(dense_docs, sparse_docs)
        return self.rerank(merged, query)

    @staticmethod
    def run():
        """
        Factory method to instantiate a Retrieval pipeline.

        Loads pre-chunked SciBERT dataset and constructs
        both dense and sparse retrievers.

        Returns
        -------
        Retrieval
            Initialized Retrieval object ready for querying.
        """
        
        chunked_dataset = SciBERTChunker.load_articles(CHUNKED_JSON)
        corpus_sparse = [chunk["text"] for chunk in chunked_dataset]
        return Retrieval(sparse_corpus=corpus_sparse)

if __name__ =="__main__":

    retriever=print(Retrieval().retrieve(query="How Quantization work"))
    print("hello")