
import json
from langchain.text_splitter import MarkdownTextSplitter
import spacy


nlp = spacy.load("en_core_web_sm") 

def load_articles(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    return articles



def chunk_articles_markdown(articles, chunk_size=1000, chunk_overlap=200):
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_dataset = []
    for article in articles:
        metadata_str = f"{article['metadata']}" if 'metadata' in article else ""
        full_text = metadata_str + "\n" + (article.get("text") or "")
        chunks = splitter.split_text(full_text)
        chunked_dataset.extend(
            {
                "source_id": article.get("metadata", {}).get("source", ""),
                "metadata": article.get("metadata", {}),
                "chunk_id": idx,
                "text": chunk,
            }
            for idx, chunk in enumerate(chunks)
        )
    return chunked_dataset

def save_chunked_dataset(chunked_dataset, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(chunked_dataset, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    json_path = "arxiv_papers/extracted_content.json"
    articles = load_articles(json_path)
    chunked_data = chunk_articles_markdown(articles)
    save_chunked_dataset(chunked_data, "arxiv_papers/chunked_dataset.json")