import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langsmith import Client
from langsmith.run_helpers import trace
from RAG.generator import Generator
from RAG.retriever import Retrieval


class Evaluation:
    """
    Full RAG (Retrieval-Augmented Generation) evaluation pipeline.

    This class automates the process of evaluating a RAG system's performance by:
      1. Retrieving relevant context documents
      2. Generating answers using a language model
      3. Computing RAGAS evaluation metrics
      4. Logging results to LangSmith
      5. Optionally displaying a quick dashboard summary

    Attributes
    ----------
    retriever : Retrieval
        The retrieval component combining FAISS and BM25 retrieval.
    generator : Generator
        The language model component responsible for generating answers.
    client : Client
        LangSmith client used for logging experiment results.

    Example
    -------
    >>> test_data = [
    ...     {"query": "What is overfitting?", "expected": "Overfitting happens when a model memorizes training data."},
    ...     {"query": "Define backpropagation.", "expected": "An algorithm to update neural network weights."}
    ... ]
    >>> evaluator = Evaluation()
    >>> df = evaluator.run(test_data)
    >>> print(df.head())
    """

    def __init__(
        self
    ):
        """
        Initialize the evaluation pipeline with a retriever, generator, and LangSmith client.

        Parameters
        ----------
 
        """
        # Initialize hybrid retriever and generator
        self.retriever = Retrieval.run()
        self.generator = Generator()
        self.client = Client()

    # -------------------------------------------------
    # Dataset building
    # -------------------------------------------------
    def build_dataset(self, test_set: list) -> Dataset:
        """
        Build a HuggingFace Dataset for RAG evaluation.

        This method:
          - Iterates through test queries and expected answers
          - Retrieves relevant context using the retriever
          - Generates an answer using the generator
          - Structures the data into a Dataset suitable for RAGAS metrics

        Parameters
        ----------
        test_set : list
            A list of dictionaries, each containing:
              - "query": str
              - "expected": str
            Optionally, pre-existing "question" and "answer" fields may be reused.

        Returns
        -------
        Dataset
            A HuggingFace Dataset with fields:
              ["question", "contexts", "ground_truth", "answer"]

        Raises
        ------
        KeyError
            If a test item does not contain the required keys.
        """
        records = []

        for item in test_set:
            # If dataset already formatted, reuse directly
            if "question" in item and "answer" in item:
                records.append(item)
                continue

            # Validate the minimal required structure
            if "query" not in item or "expected" not in item:
                raise KeyError(f"Each test item must contain 'query' and 'expected'. Got: {item}")

            query = item["query"]
            expected = item["expected"]

            # --- 1. Retrieve context documents ---
            context_docs = self.retriever.retrieve(query)
            context_texts = [doc.page_content for doc in context_docs]

            # --- 2. Generate model answer ---
            generated = self.generator.generate(query, context_docs)

            # --- 3. Record structured data ---
            records.append({
                "question": query,
                "contexts": context_texts,
                "ground_truth": expected,
                "answer": generated
            })

        # Convert list of dicts into a HuggingFace Dataset for RAGAS
        return Dataset.from_list(records)

    # -------------------------------------------------
    # Metrics computation
    # -------------------------------------------------
    def compute_metrics(self, dataset: Dataset) -> pd.DataFrame:
        """
        Compute RAGAS evaluation metrics on the generated dataset.

        Uses the following metrics:
          - faithfulness: factual alignment of answer to retrieved context
          - context_precision: relevance of retrieved documents
          - context_recall: completeness of retrieved context
          - answer_relevancy: semantic relevance of answer to question

        Parameters
        ----------
        dataset : Dataset
            The HuggingFace Dataset containing RAG pipeline outputs.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing metric scores and their mean.
        """
        # Evaluate with RAGAS
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, context_precision, context_recall, answer_relevancy]
        )

        # Convert RAGAS EvaluationResult to pandas DataFrame
        df = results.to_pandas()

        # Compute an average column for quick overview
        df["mean_score"] = df.mean(axis=1, numeric_only=True)
        return df

    # -------------------------------------------------
    # Logging to LangSmith
    # -------------------------------------------------
    def log_to_langsmith(self, df: pd.DataFrame) -> None:
        """
        Log evaluation results to LangSmith for experiment tracking.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing computed metrics.
        """
        with trace(name="RAG Evaluation") as t:
            t.add_metadata({"evaluation_results": df.to_dict(orient="records")})

    # -------------------------------------------------
    # Visualization
    # -------------------------------------------------
    def visualize_dashboard(self, df: pd.DataFrame) -> None:
        """
        Simple console visualization of evaluation results.

        This method can later be extended into a full dashboard visualization.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing computed metrics.
        """
        print("📊 Evaluation results preview:")
        print(df.head())

    # -------------------------------------------------
    # Full pipeline execution
    # -------------------------------------------------
    def run(self, test_set: list) -> pd.DataFrame:
        """
        Execute the full evaluation pipeline:
          1. Build dataset from test queries
          2. Compute RAGAS metrics
          3. Log to LangSmith
          4. Print dashboard summary

        Parameters
        ----------
        test_set : list
            List of test samples containing 'query' and 'expected' answers.

        Returns
        -------
        pd.DataFrame
            DataFrame with metric results for each test item.
        """
        dataset = self.build_dataset(test_set)
        df = self.compute_metrics(dataset)
        self.log_to_langsmith(df)
        self.visualize_dashboard(df)
        return df





