import json
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

def load_articles(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def chunk_text_with_scibert(text, chunk_size=1000, chunk_overlap=200):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
        start += chunk_size - chunk_overlap
    return chunks

def chunk_articles_with_scibert(articles, chunk_size=1000, chunk_overlap=200):
    chunked_dataset = []
    for article in articles:
        metadata_str = f"{article['metadata']}" if 'metadata' in article else ""
        full_text = metadata_str + "\n" + (article.get("text") or "")
        chunks = chunk_text_with_scibert(full_text, chunk_size, chunk_overlap)
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
    chunked_data = chunk_articles_with_scibert(articles)
    save_chunked_dataset(chunked_data, "arxiv_papers/chunked_dataset_scibert.json")

