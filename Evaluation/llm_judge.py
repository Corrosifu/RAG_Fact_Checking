import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class LLMJudge:
    """
    Lightweight evaluator using FLAN-T5-small (free & CPU-friendly).
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        device: str = None,
        max_new_tokens: int = 256
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📦 Loading judge model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.max_new_tokens = max_new_tokens
        print("✅ Judge model loaded successfully.")

    def evaluate(self, question: str, generated_answer: str, reference_answer: str) -> str:
        """
        Prompt the judge to give a short evaluation and score out of 10.
        """
        prompt = f"""
Evaluate the generated answer compared to the reference.

Criteria:
- Relevance
- Faithfulness
- Completeness
- Fluency

Question: {question}
Generated: {generated_answer}
Reference: {reference_answer}

Give a short evaluation and a score out of 10.
"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

