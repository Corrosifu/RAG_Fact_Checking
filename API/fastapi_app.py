from fastapi import FastAPI
from pydantic import BaseModel
from RAG.rag import RAG 

app = FastAPI(title="RAG API")


rag = RAG()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"status": "🚀 API is running!"}


@app.post("/query")
def ask_question(payload: QueryRequest):
    query = payload.query
    print("Received query:", query)  # for debugging
    try:
        answer = rag.ask(query)  # call your RAG method
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
