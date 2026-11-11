"""Streamlit front-end.

Run: ``BACKEND_URL=http://localhost:8000 streamlit run ui/streamlit_app.py``.
"""

import streamlit as st

from rag_fact_checking.client import BackendError, post

st.set_page_config(page_title="RAG Fact-Checking", page_icon="🔬")
st.title("🔬 Scientific RAG Fact-Checking")

tab_ask, tab_check = st.tabs(["Ask a question", "Fact-check a claim"])

STANCE_STYLE = {
    "supports": ("✅", "green", "Supports the claim"),
    "contradicts": ("❌", "red", "Contradicts the claim"),
    "neutral": ("⚪", "gray", "Neutral"),
}


def show_sources(sources: list[dict], label: str = "Sources") -> None:
    st.caption(label)
    for i, s in enumerate(sources, start=1):
        with st.expander(f"[{i}] {s['title'] or s['doc_id']} · score {s['score']}"):
            st.write(s["snippet"])


def show_sentence(s: dict) -> None:
    icon, color, label = STANCE_STYLE[s["stance"]]
    st.markdown(f"{icon} :{color}[**{label}**] · {s['title'] or s['doc_id']} · `{s['chunk_id']}`")
    st.markdown(f"> {s['text']}")
    st.caption(
        f"entailment {s['entailment']:.0%} · contradiction {s['contradiction']:.0%} "
        f"· neutral {s['neutral']:.0%}"
    )


def show_evidence_sentences(sentences: list[dict]) -> None:
    """The exact sentences the NLI model judged, strongest signal first."""
    if not sentences:
        st.info("No sentence in the retrieved passages is related enough to the claim.")
        return
    decisive = [s for s in sentences if s["stance"] != "neutral"]
    neutral = [s for s in sentences if s["stance"] == "neutral"]
    st.markdown("#### Evidence sentences")
    for s in decisive:
        show_sentence(s)
    if not decisive:
        st.caption("No sentence clearly supports or contradicts the claim.")
    if neutral:
        with st.expander(f"Neutral sentences ({len(neutral)})"):
            for s in neutral:
                show_sentence(s)


with tab_ask:
    question = st.text_input("Question")
    if st.button("Ask") and question:
        try:
            with st.spinner("Searching…"):
                data = post("/query", {"query": question})
            st.markdown(data["text"])
            show_sources(data["sources"])
        except BackendError as exc:
            st.error(str(exc))

with tab_check:
    claim = st.text_input("Claim")
    if st.button("Check") and claim:
        try:
            with st.spinner("Verifying…"):
                data = post("/fact-check", {"claim": claim})
            st.subheader(f"{data['verdict']} (confidence {data['confidence']:.0%})")
            show_evidence_sentences(data["evidence_sentences"])
            show_sources(data["evidence"], "Retrieved passages")
        except BackendError as exc:
            st.error(str(exc))
