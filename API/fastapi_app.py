# fastapi_app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from RAG.rag import RAG  # adjust to your RAG class path

app = FastAPI(title="RAG API")

# Initialize your RAG instance (load retriever + generator)
rag = RAG()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def ask_question(payload: QueryRequest):
    query = payload.query
    print("Received query:", query)  # for debugging
    try:
        answer = rag.ask(query)  # call your RAG method
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
