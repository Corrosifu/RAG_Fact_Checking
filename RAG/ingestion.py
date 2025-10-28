import os
import requests
import feedparser
import json
import pdfplumber
import camelot.io as camelot
import pymupdf4llm
from config import AXRIV_PDF,DATA_DIR,EXTRACTED_JSON,METADATA_JSON
# Base URL for querying arXiv API
ARXIV_API_URL = "http://export.arxiv.org/api/query"


class ArxivIngestor:
    """
    A class to ingest scientific papers from arXiv, download PDFs,
    extract text/markdown content, and save datasets locally.
    
    Attributes:
        base_dir (str): Directory where PDFs and extracted content are saved.
        topic (str): Search topic for arXiv query.
        max_results (int): Maximum number of articles to fetch from arXiv.
    """

    def __init__(self, topic: str="machine learning" , max_results: int=10):
        """
        Initializes the ArxivIngestor with the specified base directory, topic, and max results.
        Creates the base directory if it doesn't exist.
        """
        self.base_dir = AXRIV_PDF
        self.topic = topic
        self.max_results = max_results
        os.makedirs(self.base_dir, exist_ok=True)

    def fetch_metadata(self)->list[dict]:
        """
        Fetches metadata for articles matching the topic from the arXiv API.
        
        Returns:
            List[dict]: List of article metadata dictionaries containing id, title,
                        authors, summary, published date, and PDF URL.
        """
        query_str = self.topic.replace(" ", "+")
        url = f"{ARXIV_API_URL}?search_query=all:{query_str}&start=0&max_results={self.max_results}&sortBy=submittedDate&sortOrder=descending"
        feed = feedparser.parse(url)

        articles = [
            {
                "id": entry.id.split('/abs/')[-1],
                "title": entry.title,
                "authors": [author.name for author in entry.authors],
                "summary": entry.summary,
                "published": entry.published,
                "pdf_url": entry.links[1].href,  # PDF link
            }
            for entry in feed.entries
        ]

        print(f"Fetched metadata for {len(articles)} articles on '{self.topic}'")
        return articles

    def download_pdf(self, pdf_url:str, filepath:str):
        """
        Downloads a PDF from a given URL and saves it locally.
        
        Args:
            pdf_url (str): URL of the PDF to download.
            filepath (str): Local file path where the PDF will be saved.
        """
        if os.path.exists(filepath):
            print(f"PDF already exists: {filepath}")
            return
        response = requests.get(pdf_url)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {filepath}")

    def download_all_pdfs(self, articles:list[dict]):
        """
        Downloads PDFs for all articles and saves the metadata as JSON.
        
        Args:
            articles (List[dict]): List of article metadata dictionaries.
        """
        for article in articles:
            pdf_filename = article["id"].replace("/", "_") + ".pdf"
            pdf_path = os.path.join(self.base_dir, pdf_filename)
            try:
                self.download_pdf(article["pdf_url"], pdf_path)
            except Exception as e:
                print(f"Error downloading {article['pdf_url']}: {e}")

        # Save metadata
        metadata_file = METADATA_JSON
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"Saved metadata to {metadata_file}")

    def load_existing_dataset(self)->list[dict]:
        """
        Loads previously extracted content if it exists.
        
        Returns:
            List[dict]: Previously extracted dataset, or empty list if none exists.
        """
        path = EXTRACTED_JSON
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def extract_text(self, pdf_path:str)->str:
        """
        Extracts raw text from a PDF using pdfplumber.
        
        Args:
            pdf_path (str): Path to the PDF file.
        
        Returns:
            str: Extracted text from all pages of the PDF.
        """
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        return text

    def extract_md(self, pdf_path:str)->str:
        """
        Converts a PDF to Markdown format using pymupdf4llm.
        
        Args:
            pdf_path (str): Path to the PDF file.
        
        Returns:
            str: Markdown-formatted content of the PDF.
        """
        return pymupdf4llm.to_markdown(pdf_path)

    def extract_all_content(self, articles:list[dict])->list[dict]:
        """
        Extracts content from all PDFs, skipping already processed articles.
        
        Args:
            articles (List[dict]): List of article metadata.
        
        Returns:
            List[dict]: List of content dictionaries containing metadata and extracted text.
        """
        existing_dataset = self.load_existing_dataset()
        existing_ids = {item['metadata']['id'] for item in existing_dataset}
        dataset = existing_dataset.copy()

        for article in articles:
            if article['id'] in existing_ids:
                continue  # skip already processed articles

            pdf_filename = article["id"].replace("/", "_") + ".pdf"
            pdf_path = os.path.join(self.base_dir, pdf_filename)

            if not os.path.exists(pdf_path):
                print(f"Missing PDF: {pdf_path}")
                continue

            print(f"Extracting PDF content: {pdf_path}")
            content = {
                "metadata": article,
                "text": self.extract_md(pdf_path),
                # Placeholder for tables/images if needed
                # "tables": self.extract_tables(pdf_path),
                # "images": self.extract_images(pdf_path)
            }
            dataset.append(content)

        return dataset

    def save_dataset(self, dataset:list[dict], filename:str=None):
        """
        Saves the extracted dataset to a JSON file.
        
        Args:
            dataset (List[dict]): Extracted content dataset.
            filename (str, optional): File path to save the dataset. Defaults to "extracted_content.json".
        """
        filename = EXTRACTED_JSON
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"Dataset saved to {filename}")

    def run(self)->list[dict]:
        """
        Full ingestion pipeline:
        1. Fetch metadata from arXiv
        2. Download PDFs
        3. Extract content (Markdown)
        4. Save dataset
        
        Returns:
            List[dict]: Final dataset containing metadata and extracted content.
        """
        articles = self.fetch_metadata()
        self.download_all_pdfs(articles)
        dataset = self.extract_all_content(articles)
        self.save_dataset(dataset)
        return dataset


if __name__ == "__main__":
    ingestor = ArxivIngestor(topic="machine learning", max_results=10)
    dataset = ingestor.run()
    print(f"Total articles in dataset: {len(dataset)}")
