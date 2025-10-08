import os
import torch
from sentence_transformers import SentenceTransformer
from Chunking import load_articles
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True"

def vectorstore(persist_dir, embedding_model):
    data = load_articles("arxiv_papers/chunked_dataset.json")
    texts = [item["text"] for item in data]
    print(texts[1])
    print(texts[2])
    """metadatas = [item["metadata"] for item in data]  # Should be a list of dicts

    def embedding_function(texts):
        return embedding_model.embed_documents(texts)

    vectorstore = Chroma(
        collection_name="machine_learning",
        persist_directory=persist_dir
    )    
    for text in texts:
        vectorstore.add_texts(texts=embedding_function(text), metadatas=metadatas)
        vectorstore.persist()
    print(f"Indexation terminée et sauvegardée dans {persist_dir}")"""

if __name__ == "__main__":
    model = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}  # optional
    )
    vectorstore(persist_dir="vectorstore_dir", embedding_model=model)
