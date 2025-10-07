
import json
from langchain.text_splitter import SpacyTextSplitter
import spacy


nlp = spacy.load("en_core_web_sm") 

def load_articles(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    return articles
