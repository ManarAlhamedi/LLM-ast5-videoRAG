# The Streamlit App 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from interface.retriever import Retriever
from interface.ui_utils import load_keyframe, find_frame_for_text

st.set_page_config(page_title="🎥 RAG Video QA", layout="wide")

st.title("🎬 RAG Video QA App")

# Sidebar Settings
st.sidebar.header("Settings")
retrieval_mode = st.sidebar.selectbox("Retrieval Mode", ["semantic", "lexical"])
backend = st.sidebar.selectbox(
    "Backend",
    ["faiss", "pgvector"] if retrieval_mode == "semantic" else ["tfidf", "bm25"]
)
top_k = st.sidebar.slider("Top-k Results", 1, 10, 5)

# Instantiate retriever
retriever = Retriever(mode=retrieval_mode, backend=backend)

# Main Input
query = st.text_input("Ask a question about the video...")

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching..."):
            results = retriever.search(query, top_k=top_k)

        for idx, (text, _) in enumerate(results):
            st.markdown(f"### 🎯 Match {idx+1}")
            st.write(text)

            frame_name, timestamp = find_frame_for_text(text)
            if frame_name:
                frame = load_keyframe(frame_name)
                st.image(frame, caption=f"Frame at {timestamp:.2f} seconds")
            else:
                st.write("*No matching frame found.*")

