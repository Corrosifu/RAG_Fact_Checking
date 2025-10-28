from RAG.generator import Generator
from RAG.retriever import Retrieval


class RAG :
    
    """RAG class calling retriever and generator to process the rag action"""
    def __init__(self):

        self.retriever=Retrieval()
        self.generator=Generator() 
    def ask(self, query:str, max_tokens:int=256)->str:

        context = self.retriever.retrieve(query)
        return self.generator.generate(query, context, max_tokens=max_tokens)