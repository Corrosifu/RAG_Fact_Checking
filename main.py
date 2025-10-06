import os
import json
from Ingestion import pdf_ingestion

TOPIC = "machine learning"
BASE_DOWNLOAD_DIR = "arxiv_papers"
MAX_RESULTS = 100

def main():
   pdf_ingestion()


if __name__ == "__main__":
    main()
