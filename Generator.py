import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import Retrieval

class Generator:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct", device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          torch_dtype=torch.float16,
                                                          device_map="auto" if device == "cuda" else None)
    def format_prompt(self, query, context):
     
        context_text = "\n".join([item.page_content for item in context])
        return f"Answer the following question using the context below:\nContext:\n{context_text}\nQuestion: {query}\nAnswer:"

    def generate(self, query, context, max_tokens=256):
        prompt = self.format_prompt(query, context)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens)
            
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return response.split("Answer:")[-1].strip()

if __name__=='__main__':
    import torch
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    print("GPU memory cleared")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MiB")

    query = "How does RAG work?"
    generator = Generator()
    retriever = Retrieval.main()
    context = retriever.retrieve(query)
    answer = generator.generate(query, context)
    print("Answer:", answer)
