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
        def entailment_probs(self, claim, premises):
            return [{"entailment": 0.95, "contradiction": 0.01, "neutral": 0.04} for _ in premises]

    p = RAGPipeline(retriever, generator=None, verifier=Stub())
    res = p.fact_check("Quantization makes models smaller.")
    assert res.verdict is Verdict.SUPPORTED and res.confidence == 0.95
    assert res.to_dict()["verdict"] == "SUPPORTED" and res.evidence
    # the exact sentences used as proof are returned with their source and probabilities
    assert res.evidence_sentences and all(e.stance == "supports" for e in res.evidence_sentences)
    first = res.to_dict()["evidence_sentences"][0]
    assert first["doc_id"] == "quantization" and first["entailment"] == 0.95
    assert first["text"] and first["chunk_id"].startswith("quantization#")


def test_from_settings_end_to_end(settings, sample_corpus):
    from rag_fact_checking.corpus import build_index

    build_index(settings)
    ans = RAGPipeline.from_settings(settings).ask("What is LoRA fine-tuning?")
    assert ans.sources[0].doc_id == "lora"


class FixedGenerator:
    def __init__(self, text):
        self.text = text

    def generate(self, query, hits, max_new_tokens=256):
        return self.text


ANSWER = "Quantization reduces the memory footprint of neural networks using fewer bits."


def test_ask_verifies_each_sentence_when_verifier_configured(retriever):
    from stubs import StubVerifier

    p = RAGPipeline(retriever, FixedGenerator(ANSWER), verifier=StubVerifier(), top_k=3)
    ans = p.ask("What does quantization do?")
    assert [c.status for c in ans.checks] == ["supported"]
    assert ans.to_dict()["checks"][0]["evidence"]["doc_id"] == "quantization"


def test_verification_can_be_disabled_per_call_or_globally(retriever):
    from stubs import StubVerifier

    p = RAGPipeline(retriever, FixedGenerator(ANSWER), verifier=StubVerifier())
    assert p.ask("What does quantization do?", verify=False).checks == []
    assert (
        RAGPipeline(
            retriever, FixedGenerator(ANSWER), verifier=StubVerifier(), verify_answers=False
        )
        .ask("What does quantization do?")
        .checks
        == []
    )
    # per-call override wins over the global default
    assert (
        RAGPipeline(
            retriever, FixedGenerator(ANSWER), verifier=StubVerifier(), verify_answers=False
        )
        .ask("What does quantization do?", verify=True)
        .checks
    )


def test_no_verification_without_verifier_or_for_the_no_evidence_fallback(retriever):
    from rag_fact_checking.generation import NO_EVIDENCE
    from stubs import StubVerifier

    assert RAGPipeline(retriever, FixedGenerator(ANSWER)).ask("quantization?").checks == []
    p = RAGPipeline(retriever, FixedGenerator(NO_EVIDENCE), verifier=StubVerifier())
    assert p.ask("quantization?").checks == []
