import os
import json
from Ingestion import pdf_ingestion,save_dataset,extract_all_content


def main():
   pdf_ingestion(BASE_DOWNLOAD_DIR,TOPIC,MAX_RESULTS)
    

if __name__ == "__main__":
    TOPIC = "machine learning"
    BASE_DOWNLOAD_DIR = "arxiv_papers"
    MAX_RESULTS = 10
    main()
    metadata_path = os.path.join(BASE_DOWNLOAD_DIR, "metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    full_dataset = extract_all_content(articles, BASE_DOWNLOAD_DIR)
    save_dataset(full_dataset, os.path.join(BASE_DOWNLOAD_DIR, "extracted_content.json"))
