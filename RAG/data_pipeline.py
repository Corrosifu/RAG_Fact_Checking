from RAG.ingestion import ArxivIngestor
from RAG.chunking import SciBERTChunker
from RAG.embedding import Embedder
import os
import json


class DataPipeline:
    """
    End-to-end pipeline for scientific paper ingestion, processing, and vectorization.

    This pipeline handles:
        1. Ingesting papers from arXiv using a keyword/topic.
        2. Chunking full-text documents into smaller segments.
        3. Embedding chunks and storing them in a FAISS vector index.

    Attributes:
        base_dir (str): Directory to store downloaded papers and processed data.
        topic (str): Topic keyword to search in arXiv.
        max_results (int): Maximum number of papers to ingest.
        ingestor (ArxivIngestor): Handles fetching metadata and PDFs from arXiv.
        chunker (SciBERTChunker): Handles tokenization and chunking of full-text PDFs.
        embedder (Embedder): Handles embedding chunks and building the FAISS index.
    """

    def __init__(self, topic="machine learning", max_results=10):
        """
        Initialize the data pipeline with optional parameters.

        Args:
            topic (str, optional): Topic for querying arXiv.
            max_results (int, optional): Number of papers to retrieve.
        """
        
        self.topic = topic
        self.max_results = max_results

        # Instantiate each stage of the pipeline
        self.ingestor = ArxivIngestor(topic, max_results)
        self.chunker = SciBERTChunker()
        self.embedder = Embedder()

    def run(self):
        """
        Execute the full data pipeline sequentially.

        Steps:
            1. Fetch metadata and download PDFs from arXiv.
            2. Chunk PDFs into manageable text segments.
            3. Embed chunks and store embeddings in FAISS.

        Returns:
            dict: Paths to key outputs (metadata, extracted content, chunked dataset, FAISS index).
        """
        print("Starting full data pipeline...")

        # Step 1: Ingest papers
        print("Ingesting data...")
        self.ingestor.run()

        # Step 2: Chunking PDFs
        print("Chunking data...")
        self.chunker().run()

        # Step 3: Embedding and FAISS indexing
        print("Embedding and Vector Storing...")
        self.embedder().run()

        print("✅ Data pipeline completed successfully!")

        return {
            "metadata_path": "data/metadata.json",
            "extracted_path": "data/extracted_content.json",
            "chunked_path": "data/chunked_dataset_scibert.json",
            "faiss_index": "data/faiss_index"
        }


if __name__ == "__main__":
    # Run the pipeline directly
    data_pipeline = DataPipeline()
    data_pipeline.run()
