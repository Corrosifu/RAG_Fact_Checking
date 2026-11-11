"""Optional LLM-based evaluation with RAGAS (``pip install .[eval]``).

Experimental: needs an LLM judge, so it is heavy and not part of the CI. Use
:mod:`rag_fact_checking.evaluation.metrics` for fast, deterministic retrieval metrics.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_fact_checking.pipeline import RAGPipeline


def build_ragas_dataset(pipeline: RAGPipeline, testset: Sequence[dict]):
    from ragas import EvaluationDataset

    records = []
    for item in testset:
        hits = pipeline.retriever.retrieve(item["query"], pipeline.top_k)
        answer = pipeline.generator.generate(item["query"], hits, pipeline.max_new_tokens)
        records.append(
            {
                "user_input": item["query"],
                "retrieved_contexts": [h.chunk.text for h in hits],
                "response": answer,
                "reference": item.get("expected", ""),
            }
        )
    return EvaluationDataset.from_list(records)


def run_ragas(pipeline: RAGPipeline, testset: Sequence[dict], llm, embeddings):
    """Run faithfulness / answer relevancy / context precision / context recall."""
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    result = evaluate(
        dataset=build_ragas_dataset(pipeline, testset),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    return result.to_pandas()
