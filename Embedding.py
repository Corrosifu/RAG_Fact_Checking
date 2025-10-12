from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import torch
import os
from Chunking import SciBERTChunker

# Configure PyTorch CUDA memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True"


class Embedder:
    """
    Handles embedding of scientific articles and building a FAISS vector index.

    Attributes:
        device (str): Device used for embeddings ('cuda' or 'cpu').
        embedder (HuggingFaceEmbeddings): HuggingFace embeddings model instance.
        batch_size (int): Number of documents to embed at once.
    """

    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B", device=None, batch_size=1):
        """
        Initializes the Embedder with a HuggingFace embedding model.

        Args:
            model_name (str): Name of the HuggingFace embedding model.
            device (str, optional): Device to run embeddings on; defaults to GPU if available.
            batch_size (int, optional): Number of documents processed in a batch.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size}
        )
        self.batch_size = batch_size

    def embed_docs(self, chunked_dataset):
        """
        Converts text chunks into embeddings and wraps them as Document objects.

        Args:
            chunked_dataset (List[Dict]): Chunked dataset with 'text' and 'metadata' fields.

        Returns:
            Tuple[List[Document], List[np.ndarray]]: List of Document objects and their embeddings.
        """
        texts = [chunk["text"] for chunk in chunked_dataset]
        metadatas = [chunk["metadata"] for chunk in chunked_dataset]

        # Wrap text + metadata into Document objects
        docs = [Document(page_content=txt, metadata=meta) for txt, meta in zip(texts, metadatas)]

        embeddings = []

        # Embed documents in batches to avoid memory issues
        for i in tqdm(range(0, len(docs), self.batch_size), desc="Embedding"):
            batch_docs = docs[i:i+self.batch_size]
            batch_texts = [doc.page_content for doc in batch_docs]
            batch_embeddings = self.embedder.embed_documents(batch_texts)
            embeddings.extend(batch_embeddings)
            torch.cuda.empty_cache()  # Clear GPU memory after each batch

        return docs, embeddings

    def build_faiss_index(self, docs, embeddings, index_path="faiss_index"):
        """
        Builds a FAISS vector store from document embeddings and saves it locally.

        Args:
            docs (List[Document]): List of Document objects.
            embeddings (List[np.ndarray]): Corresponding embeddings.
            index_path (str, optional): Path to save the FAISS index.

        Returns:
            FAISS: Built FAISS vector store.
        """
        faiss_db = None
        with tqdm(total=len(docs), desc="Indexing FAISS") as pbar:
            for doc, embedding in zip(docs, embeddings):
                if faiss_db is None:
                    # Initialize FAISS vector store with the first document
                    faiss_db = FAISS.from_documents([doc], self.embedder)
                else:
                    # Add documents incrementally
                    faiss_db.add_documents([doc], embeddings=[embedding])
                pbar.update(1)

        # Save FAISS index for later retrieval
        faiss_db.save_local(index_path)
        return faiss_db

    def run(self, chunked_dataset_path="arxiv_papers/chunked_dataset_scibert.json", index_path="faiss_index"):
        """
        Complete pipeline to load a chunked dataset, embed all documents, and build a FAISS index.

        Args:
            chunked_dataset_path (str, optional): Path to the JSON file containing chunked dataset.
            index_path (str, optional): Path to save the FAISS index.

        Returns:
            FAISS: Built FAISS vector store.
        """
        chunked_dataset = SciBERTChunker.load_articles(chunked_dataset_path)
        docs, embeddings = self.embed_docs(chunked_dataset)
        return self.build_faiss_index(docs, embeddings, index_path=index_path)


if __name__ == "__main__":
    # Instantiate the embedder and build the FAISS vector store
    embedder = Embedder(batch_size=1)
    vectorstore = embedder.run()

