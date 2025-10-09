from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from Chunking import load_articles
import os
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True"


def embed_vectorstore(chunked_dataset):
    
    embedder = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True,"batch_size":2}  
    )
    texts = [chunk["text"] for chunk in chunked_dataset]
    metadatas = [chunk["metadata"] for chunk in chunked_dataset]
    embeddings=[]
    batch_size = 2  
    docs = [Document(page_content=txt, metadata=meta) for txt, meta in zip(texts, metadatas)]
    


    for i in tqdm(range(0, len(docs), batch_size), desc="Embedding"):
        batch_docs = docs[i:i+batch_size]
        batch_texts = [doc.page_content for doc in batch_docs]
        batch_embeddings = embedder.embed_documents(batch_texts)
        embeddings.extend(batch_embeddings)
        torch.cuda.empty_cache()

    faiss_db = None
    with tqdm(total=len(docs), desc="Indexing FAISS") as pbar:
        for doc, embedding in zip(docs, embeddings):
            if faiss_db is None:
                faiss_db = FAISS.from_documents([doc], embedder)
            else:
                faiss_db.add_documents([doc], embeddings=[embedding])
            pbar.update(1)

    faiss_db.save_local("faiss_index")



if __name__ == "__main__":

    chunked_dataset_path = "arxiv_papers/chunked_dataset_scibert.json"
    chunked_dataset = load_articles(chunked_dataset_path)
    vectorstore = embed_vectorstore(chunked_dataset)

