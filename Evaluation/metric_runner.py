import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import EvaluationDataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ragas.run_config import RunConfig

METRIC_MAP = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}

class MetricRunner:
    def __init__(self, metrics=None, device: str = "cpu", max_total_tokens: int = 4024):
        self.metrics = [METRIC_MAP[m] for m in (metrics or METRIC_MAP.keys())]
        self._hf_device = 0 if device.startswith("cuda") else -1
        self.max_total_tokens = max_total_tokens  # Reserve space for generation

    def _build_embeddings(self):
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def _build_llm(self):
        model_id = "microsoft/Phi-3-mini-128k-instruct-onnx"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(model_id,trust_remote_code=True)
        gen = pipeline(
            "text-generation",
            model=mdl,
            tokenizer=self.tok,
            device=self._hf_device,
            max_new_tokens=128,
            do_sample=False,
        )
        return HuggingFacePipeline(pipeline=gen)

    """def _truncate_text(self, text, max_tokens):
        tokens = self.tok.tokenize(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]  # keep beginning tokens
        return self.tok.convert_tokens_to_string(tokens)

    def _prepare_dataset(self, dataset):
        # Truncate combined tokens for contexts + answer so total < max_total_tokens
        def truncate_row(row):
            question = row["question"]
            answer = row["answer"]
            contexts = row["contexts"]
            # Combine contexts as single string (or adjust as needed)
            combined_context = " ".join(contexts) if isinstance(contexts, list) else contexts
            
            # Tokenize question, context, answer to count tokens
            q_tokens = len(self.tok.tokenize(question))
            c_tokens = len(self.tok.tokenize(combined_context))
            a_tokens = len(self.tok.tokenize(answer))

            total_len = q_tokens + c_tokens + a_tokens
            if total_len <= self.max_total_tokens:
                # no truncation needed
                return {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                }

            # allocate tokens proportionally (simple heuristic)
            excess = total_len - self.max_total_tokens
            # Prioritize question with no truncation, proportionally trim contexts and answer
            max_c_tokens = max(c_tokens - excess//2, 20)  # minimum tokens to keep significant context
            max_a_tokens = max(a_tokens - excess//2, 10)  # minimum tokens for answer

            truncated_context = self._truncate_text(combined_context, max_c_tokens)
            truncated_answer = self._truncate_text(answer, max_a_tokens)

            # Return truncated with minimal token loss
            return {
                "question": question,
                "answer": truncated_answer,
                "contexts": [truncated_context],  # contexts here as single string in list
            }

        return dataset.map(truncate_row)"""

    def run(self, dataset):
        embeddings = self._build_embeddings()
        llm = self._build_llm()

        #truncated_dataset = self._prepare_dataset(dataset)
        run_config = RunConfig(timeout=20000, max_workers=32)
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            embeddings=embeddings,
            llm=llm,
            run_config=run_config
        )
        df = results.to_pandas()
        df["mean_score"] = df.select_dtypes("number").mean(axis=1)
        return df
