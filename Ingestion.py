import os
import requests
import feedparser
import json
import pdfplumber
import camelot.io as camelot
#import fitz
import pymupdf4llm
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

def extract_md(pdf_path):
    return pymupdf4llm.to_markdown(pdf_path)

"""def extract_tables(pdf_path):
    tables = camelot.read_pdf(pdf_path, pages='all')
    return [table.df for table in tables]"""



"""def extract_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png")
            images.append({
                "xref": xref,
                "image_bytes": image_bytes,
                "ext": image_ext
            })
    doc.close()
    return images"""


def extract_all_content(articles, base_dir):

    dataset = []
    for article in articles:
        pdf_filename = article["id"].replace('/', '_') + ".pdf"
        pdf_path = os.path.join(base_dir, pdf_filename)
        if not os.path.exists(pdf_path):
            print(f"PDF manquant: {pdf_path}")
            continue

        print(f"Extraction contenu PDF: {pdf_path}")
        content = {
            "metadata": article,
            "text": extract_md(pdf_path),
            #"tables": extract_tables(pdf_path),
            #"images": extract_images(pdf_path),  # image bytes, à gérer plus tard
        }
        dataset.append(content)

    return dataset


def save_dataset(dataset, filename):
    dataset_filtered = []
    for item in dataset:
        filtered_item = item.copy()

        # Convertir les DataFrames en listes de dictionnaires
        """filtered_item["tables"] = []
        for table in item.get("tables", []):
            # Si table est un DataFrame
            if hasattr(table, "to_dict"):
                filtered_item["tables"].append(table.to_dict(orient="records"))
            else:
                filtered_item["tables"].append(table)

        # Gérer les images (bytes en string base64)
        filtered_images = []
        for img in item.get("images", []):
            if isinstance(img, dict):
                import base64
                # Encoder les bytes en base64 pour JSON
                encoded_image = base64.b64encode(img.get("image_bytes", b"")).decode('utf-8')
                filtered_images.append({
                    "xref": img.get("xref", ""),
                    "ext": img.get("ext", ""),
                    "image_base64": encoded_image
                })
        filtered_item["images"] = filtered_images"""
        dataset_filtered.append(filtered_item)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dataset_filtered, f, ensure_ascii=False, indent=2)
    print(f"Données extraites sauvegardées dans {filename}")


if __name__ == "__main__":
    TOPIC = "machine learning"
    BASE_DOWNLOAD_DIR = "arxiv_papers"
    MAX_RESULTS = 10
    pdf_ingestion(BASE_DOWNLOAD_DIR,TOPIC,MAX_RESULTS) 
    metadata_path = os.path.join(BASE_DOWNLOAD_DIR, "metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    full_dataset = extract_all_content(articles, BASE_DOWNLOAD_DIR)
    save_dataset(full_dataset, os.path.join(BASE_DOWNLOAD_DIR, "extracted_content.json"))