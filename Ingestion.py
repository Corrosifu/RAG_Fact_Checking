import os
import requests
import feedparser
import json
import pdfplumber
import camelot
import fitz

ARXIV_API_URL = "http://export.arxiv.org/api/query"

def fetch_arxiv_metadata(query, max_results=100):
    query_str = query.replace(" ", "+")
    url = f"{ARXIV_API_URL}?search_query=all:{query_str}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    feed = feedparser.parse(url)
    articles = []
    articles.extend(
        {
            "id": entry.id.split('/abs/')[-1],
            "title": entry.title,
            "authors": [author.name for author in entry.authors],
            "summary": entry.summary,
            "published": entry.published,
            "pdf_url": entry.links[1].href,  # usually 1 is PDF link
        }
        for entry in feed.entries
    )
    return articles

def download_pdf(pdf_url, filepath):
    if os.path.exists(filepath):
        print(f"PDF already exists: {filepath}")
        return
    response = requests.get(pdf_url)
    response.raise_for_status()
    with open(filepath, "wb") as f:
        f.write(response.content)
    print(f"Downloaded {filepath}")


def pdf_ingestion(BASE_DOWNLOAD_DIR,TOPIC,MAX_RESULTS):
    os.makedirs(BASE_DOWNLOAD_DIR, exist_ok=True)
    articles = fetch_arxiv_metadata(TOPIC, MAX_RESULTS)
    print(f"Fetched metadata for {len(articles)} articles on '{TOPIC}'")

    for article in articles:
        pdf_filename = article["id"].replace('/', '_') + ".pdf"
        pdf_path = os.path.join(BASE_DOWNLOAD_DIR, pdf_filename)
        try:
            download_pdf(article["pdf_url"], pdf_path)
        except Exception as e:
            print(f"Error downloading {article['pdf_url']}: {e}")

    metadata_file = os.path.join(BASE_DOWNLOAD_DIR, "metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata file to {metadata_file}")


def extract_text(pdf_path):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            all_text += page.extract_text() + "\n"
    return all_text



def extract_tables(pdf_path):
    tables = camelot.read_pdf(pdf_path, pages='all')
    return [table.df for table in tables]



def extract_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(image_bytes)  # à sauvegarder ou traiter
    return images
