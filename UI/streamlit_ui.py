# streamlit_ui.py
import streamlit as st
import requests

st.title("Scientific RAG QA")

query = st.text_input("Ask your question here:")

if st.button("Ask") and query:
    try:
        response = requests.post(
            "http://localhost:8000/query", 
            json={"query": query},
            timeout=750
        )
        if response.status_code == 200:
            data = response.json()
            if "answer" in data:
                st.markdown(f"**Answer:** {data['answer']}")
            else:
                st.error(f"Error: {data.get('error', 'Unknown error')}")
        else:
            st.error(f"Request failed with status {response.status_code}")
    except Exception as e:
        st.error(f"Exception: {str(e)}")
