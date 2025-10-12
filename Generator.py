import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True"


class Generator:
    """
    Pedagogical scientific explanation generator.
    All context tensors and inputs are moved to self.device.
    """

    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", device=None, max_context_chars=1500):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_context_chars = max_context_chars

        print(f"📦 Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        print("✅ Model loaded.")

    def _clean_context(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\(.*?\d{4}.*?\)', '', text)
        text = re.sub(r'[^a-zA-Z0-9.,;:\-\s]', '', text)
        return text.strip()

    def format_prompt(self, query, context):
        context_text = "\n".join([self._clean_context(item.page_content) for item in context])
        context_text = context_text[:self.max_context_chars]
        return (
            f"You are a friendly and knowledgeable scientific teaching assistant. "
            f"Explain complex machine learning or scientific concepts clearly and concisely, using simple language "
            f"and examples when appropriate. Do not copy text from the context; instead, explain it in your own words.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
        )

    def generate(self, query, context, max_tokens=224):
        prompt = self.format_prompt(query, context)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = output[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()
