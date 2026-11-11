"""The bibliography is not evidence: it must neither be indexed nor read by the verifier."""

import pytest

from rag_fact_checking.chunking import chunk_document
from rag_fact_checking.config import Settings
from rag_fact_checking.corpus import build_index, save_documents
from rag_fact_checking.generation import ExtractiveGenerator
from rag_fact_checking.schemas import Chunk, Document, Hit
from rag_fact_checking.textutils import (
    is_prose,
    looks_like_reference,
    reference_signals,
    strip_references,
)
from rag_fact_checking.verifier import select_premises

# Real entry that made the NLI model "contradict" a true claim (contradiction 0.92).
ENTRY = (
    "Kaczmarz algorithm with exponential convergence. Journal of Fourier Analysis and "
    "Applications , 15:262–278, 2009. Peng Wang, Shaoshuai Mou, Jianming Lian, and Wei Ren."
)
INLINE = (
    "Feed-forward Gaussian splatting (Charatan et al., 2024; Szymanowicz et al., 2025; Chen et "
    "al., 2025b) predicts the Gaussians directly from posed images."
)
BODY = " ".join(f"Sentence {i} explains the iterative method in plain words." for i in range(60))
LONG_BODY = (BODY + " ") * 4  # a paper body is longer than its reference list


@pytest.mark.parametrize(
    "heading",
    ["References", "## References", "### **References**", "**References**", "R EFERENCES",
     "REFERENCES", "Bibliography", "  ## References:  "],
)  # fmt: skip
def test_heading_variants_cut_the_reference_list(heading):
    out = strip_references(f"{BODY}\n\n\n{heading}\n\n\n[1] {ENTRY}\n[2] {ENTRY}")
    assert out.strip() == BODY and "Kaczmarz" not in out


def test_text_without_a_references_heading_is_unchanged():
    assert strip_references(BODY) == BODY
    assert (
        strip_references(f"{BODY} Their preferences were recorded.")
        == BODY + " Their preferences were recorded."
    )


def test_an_early_references_heading_is_ignored():
    text = f"References\n{BODY}\n{BODY}\n{BODY}"  # e.g. a table of contents entry
    assert strip_references(text) == text


def test_only_the_last_references_heading_counts_and_an_appendix_after_it_is_kept():
    text = f"{BODY}\n\n## References\n\n[1] {ENTRY}\n\n## Appendix A\n\nExtra proof details."
    out = strip_references(text)
    assert "Kaczmarz" not in out and out.endswith("Extra proof details.") and BODY in out


def test_a_reference_entry_is_recognised_but_inline_citations_are_not():
    assert looks_like_reference(ENTRY)
    assert reference_signals(ENTRY) >= 3
    assert not looks_like_reference(INLINE)
    assert not looks_like_reference(
        "The Kaczmarz algorithm is an iterative method for linear systems."
    )
    apa = (
        "Strohmer, T. and Vershynin, R. (2009). A randomized Kaczmarz algorithm. "
        "J. Fourier Anal. Appl. 15, 262–278."
    )
    assert not looks_like_reference("See Strohmer and Vershynin (2009) for details.")
    assert reference_signals(apa) >= 1


def test_bibliography_entries_are_not_prose():
    assert not is_prose(ENTRY)
    assert is_prose("The Kaczmarz algorithm is a row-action method used to solve linear systems.")
    assert is_prose(INLINE)  # inline citations stay: they are part of ordinary sentences


def test_chunks_exclude_the_reference_list_by_default():
    doc = Document(
        "p", f"{LONG_BODY}\n\n## References\n\n" + "\n".join(f"[{i}] {ENTRY}" for i in range(30))
    )
    kept = chunk_document(doc, size=60, overlap=10)
    assert kept and not any("Journal of Fourier" in c.text for c in kept)
    everything = chunk_document(doc, size=60, overlap=10, drop_references=False)
    assert len(everything) > len(kept) and any("Journal of Fourier" in c.text for c in everything)


def test_index_build_honours_the_setting(tmp_path):
    doc = Document(
        "p", f"{LONG_BODY}\n\n## References\n\n" + "\n".join(f"[{i}] {ENTRY}" for i in range(30))
    )
    counts = {}
    for flag in (True, False):
        s = Settings(
            data_dir=tmp_path / str(flag), chunk_size=60, chunk_overlap=10, drop_references=flag
        )
        save_documents([doc], s.corpus_path)
        counts[flag] = len(build_index(s))
    assert counts[True] < counts[False]
    assert Settings.from_env({"RAGFC_DROP_REFERENCES": "false"}).drop_references is False
    assert Settings().drop_references is True


def hit(text):
    return Hit(Chunk("d#0", "d", 0, text, "A Long Enough Document Title"), 1.0)


def test_the_verifier_never_sees_a_bibliography_entry():
    """The conflicting window of the Kaczmarz claim came from the reference list."""
    claim = "The Kaczmarz algorithm is an iterative method for solving linear systems."
    prose = (
        "One such iterative method for solving large-scale consistent linear systems is the "
        "Kaczmarz algorithm."
    )
    premises = select_premises(claim, [hit(ENTRY), hit(prose)])
    assert premises and all("Journal of Fourier" not in p.text for p in premises)
    assert select_premises(claim, [hit(ENTRY)]) == []  # nothing but references: no evidence at all


def test_extractive_answers_do_not_quote_the_bibliography():
    prose = (
        "The Kaczmarz algorithm solves linear systems by projecting onto one equation at a time."
    )
    out = ExtractiveGenerator().generate(
        "What is the Kaczmarz algorithm?", [hit(f"{ENTRY} {prose}")]
    )
    assert "Journal of Fourier" not in out and "Kaczmarz algorithm solves" in out


def test_neighbours_of_a_reference_sentence_are_dropped_too():
    from rag_fact_checking.textutils import drop_reference_neighbourhood

    sentences = [
        "A normal sentence about the method.",
        "Kaczmarz algorithm with exponential convergence.",
        "Journal of Fourier Analysis and Applications , 15:262–278, 2009.",
        "Peng Wang, Shaoshuai Mou, Jianming Lian, and Wei Ren.",
        "Another normal sentence about the results.",
    ]
    assert drop_reference_neighbourhood(sentences) == [sentences[0], sentences[4]]
    assert drop_reference_neighbourhood(sentences[:1]) == sentences[:1]
    assert drop_reference_neighbourhood([]) == []


# --- documents whose conversion lost the "References" heading -------------------------------------
ENTRIES = "\n".join(
    f"Author{i} A, Other{i} B. A study of method {i}. arXiv preprint arXiv:2401.{i:05d} , 2024."
    for i in range(60)
)
APPENDIX = " ".join(
    f"Appendix sentence {i} gives more experimental details here." for i in range(120)
)


def test_a_reference_block_is_removed_even_without_a_heading():
    text = f"{LONG_BODY}\n\n{ENTRIES}\n\n{APPENDIX}"
    out = strip_references(text)
    assert "arXiv preprint" not in out
    assert (
        "Sentence 59 explains" in out and "Appendix sentence 119" in out
    )  # prose on both sides stays


def test_prose_with_inline_citations_and_a_stray_reference_sentence_is_untouched():
    text = (INLINE + " ") * 80 + ENTRY + " " + BODY
    assert strip_references(text) == text


def test_short_documents_are_left_alone():
    assert strip_references(ENTRY) == ENTRY
