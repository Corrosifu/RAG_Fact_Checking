"""Streamlit front-end.

Run: ``BACKEND_URL=http://localhost:8000 streamlit run ui/streamlit_app.py``.
"""

import streamlit as st

from rag_fact_checking.client import BackendError, post

st.set_page_config(page_title="RAG Fact-Checking", page_icon="🔬")
st.title("🔬 Scientific RAG Fact-Checking")

tab_ask, tab_check = st.tabs(["Ask a question", "Fact-check a claim"])


def show_sources(sources: list[dict]) -> None:
    for i, s in enumerate(sources, start=1):
        with st.expander(f"[{i}] {s['title'] or s['doc_id']} · score {s['score']}"):
            st.write(s["snippet"])


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
            show_sources(data["evidence"])
        except BackendError as exc:
            st.error(str(exc))
