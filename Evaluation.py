import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langsmith import Client
from langsmith.run_helpers import trace
from Generator import Generator
from Retriever import Retrieval


class Evaluation:
    """
    RAG evaluation pipeline: builds dataset, generates answers, computes metrics, logs to LangSmith, and visualizes.
    """

    def __init__(self, faiss_index_path="faiss_index", chunked_dataset_path="arxiv_papers/chunked_dataset_scibert.json"):
        self.retriever = Retrieval.run()
        self.generator = Generator()
        self.client = Client()

    def build_dataset(self, test_set):
        """
        Convert a list of queries + expected answers into a HuggingFace Dataset with generated answers.
        """
        records = []

        for item in test_set:
            # If dataset already contains answers, reuse it
            if "question" in item and "answer" in item:
                records.append(item)
                continue

            # Validate keys
            if "query" not in item or "expected" not in item:
                raise KeyError(f"Each test item must contain 'query' and 'expected'. Got: {item}")

            query = item["query"]
            expected = item["expected"]

            # Retrieve context
            context_docs = self.retriever.retrieve(query)
            context_texts = [doc.page_content for doc in context_docs]

            # Generate answer
            generated = self.generator.generate(query, context_docs)

            records.append({
                "question": query,
                "contexts": context_texts,
                "ground_truth": expected,
                "answer": generated
            })

        return Dataset.from_list(records)

    def compute_metrics(self, dataset):
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, context_precision, context_recall, answer_relevancy]
        )

        # Convert EvaluationResult to pandas DataFrame
        df = results.to_pandas()

        # Add average score
        df["mean_score"] = df.mean(axis=1, numeric_only=True)
        return df

    def log_to_langsmith(self, df):

        with trace(name="RAG Evaluation") as t:
            t.add_metadata({"evaluation_results": df.to_dict(orient="records")})
            


    def visualize_dashboard(self, df):
        """
        Simple visualization — can be replaced by a proper dashboard later.
        """
        print("📊 Evaluation results preview:")
        print(df.head())

    def run(self, test_set):
        """
        Full evaluation pipeline.
        """
        dataset = self.build_dataset(test_set)
        df = self.compute_metrics(dataset)
        self.log_to_langsmith(df)
        self.visualize_dashboard(df)
        return df





