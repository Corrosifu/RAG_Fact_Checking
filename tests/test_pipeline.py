import pytest

from rag_fact_checking.pipeline import RAGPipeline, VerifierNotConfigured
from rag_fact_checking.schemas import Verdict


def test_ask_returns_grounded_answer_with_sources(pipeline):
    ans = pipeline.ask("What does the learning rate control in gradient descent?")
    assert "learning rate" in ans.text.lower()
    assert ans.sources[0].doc_id == "gradient-descent"
    assert len(ans.sources) <= 3 and ans.sources[0].snippet


def test_ask_top_k_override(pipeline):
    assert len(pipeline.ask("neural network", top_k=1).sources) == 1


def test_fact_check_requires_verifier(pipeline):
    with pytest.raises(VerifierNotConfigured):
        pipeline.fact_check("Quantization makes models smaller.")


def test_fact_check_with_stub_verifier(retriever):
    class Stub:
        def entailment_probs(self, claim, hits):
            return [{"entailment": 0.95, "contradiction": 0.01, "neutral": 0.04} for _ in hits]

    p = RAGPipeline(retriever, generator=None, verifier=Stub())
    res = p.fact_check("Quantization makes models smaller.")
    assert res.verdict is Verdict.SUPPORTED and res.confidence == 0.95
    assert res.to_dict()["verdict"] == "SUPPORTED" and res.evidence


def test_from_settings_end_to_end(settings, sample_corpus):
    from rag_fact_checking.corpus import build_index

    build_index(settings)
    ans = RAGPipeline.from_settings(settings).ask("What is LoRA fine-tuning?")
    assert ans.sources[0].doc_id == "lora"
