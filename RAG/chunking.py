import json
from transformers import AutoTokenizer
from config import EXTRACTED_JSON,CHUNKED_JSON

class SciBERTChunker:
    """
    A class for chunking scientific articles into smaller text segments using SciBERT tokenizer.
    Useful for creating datasets for retrieval-augmented generation or embedding-based search.

    Attributes:
        tokenizer (AutoTokenizer): HuggingFace tokenizer for SciBERT.
        chunk_size (int): Number of tokens per chunk.
        chunk_overlap (int): Number of overlapping tokens between consecutive chunks.
    """

    def __init__(self, tokenizer_name="allenai/scibert_scivocab_uncased", chunk_size=1000, chunk_overlap=200):
        """
        Initializes the SciBERTChunker with a tokenizer, chunk size, and chunk overlap.

        Args:
            tokenizer_name (str): Pretrained tokenizer name from HuggingFace.
            chunk_size (int): Number of tokens per chunk.
            chunk_overlap (int): Number of overlapping tokens between consecutive chunks.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def load_articles(json_path):
        """
        Loads articles from a JSON file.

        Args:
            json_path (str): Path to the JSON file containing articles.

        Returns:
            List[dict]: List of articles loaded from JSON.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def chunk_text(self, text):
        """
        Splits a long text into token-based chunks with optional overlap.

        Args:
            text (str): The full text of an article to chunk.

        Returns:
            List[str]: List of text chunks as strings.
        """
        # Encode text into tokens without special tokens
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0

        # Slide a window over the tokens to create overlapping chunks
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, clean_up_tokenization_spaces=True)
            chunks.append(chunk_text)
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def chunk_articles(self, articles):
        """
        Chunks all articles into smaller segments and attaches metadata.

        Args:
            articles (List[dict]): List of article dictionaries with 'metadata' and 'text'.

        Returns:
            List[dict]: List of chunked articles, each with chunk_id, source_id, metadata, and text.
        """
        chunked_dataset = []

        for article in articles:
            metadata = article.get("metadata", {})
            full_text = article.get("text", "")
            chunks = self.chunk_text(full_text)

            # Add each chunk as a separate entry with metadata
            chunked_dataset.extend(
                {
                    "source_id": metadata.get("source", ""),
                    "metadata": metadata,
                    "chunk_id": idx,
                    "text": chunk,
                }
                for idx, chunk in enumerate(chunks)
            )

        return chunked_dataset

    @staticmethod
    def save_chunked_dataset(chunked_dataset, filename):
        """
        Saves the chunked dataset to a JSON file.

        Args:
            chunked_dataset (List[dict]): List of chunked articles.
            filename (str): File path where the dataset will be saved.
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chunked_dataset, f, ensure_ascii=False, indent=2)
        print(f"Chunked dataset saved to {filename}")

    def run(self):
        """
        Full pipeline to load articles, chunk them, and save the chunked dataset.

        Args:
            json_path (str): Path to the JSON file containing full articles.
            output_file (str): Path to save the chunked dataset.

        Returns:
            List[dict]: The chunked dataset.
        """
        articles = self.load_articles(EXTRACTED_JSON)
        chunked_data = self.chunk_articles(articles)
        self.save_chunked_dataset(chunked_data, CHUNKED_JSON)
        return chunked_data


if __name__ == "__main__":
    # Instantiate the chunker and run the full pipeline
    chunker = SciBERTChunker()
    chunker.run()
