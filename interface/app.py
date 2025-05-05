import streamlit as st

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TRANSFORMERS_NO_TF'] = '1'

from retrieval.unified_retriever import UnifiedRetriever
import json


# ---------------------
# CONFIG
# ---------------------
VIDEO_PATH = "data/video.mp4"
MAPPING_PATH = "data/mappings/frame_text_mapping.json"
RETRIEVER_OPTIONS = ["bm25", "tfidf", "faiss", "pgvector", "multimodal"]

# ---------------------
# LOAD FRAME-TEXT MAPPING
# ---------------------
def load_mapping(mapping_path):
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------
# Get timestamp for result text
# ---------------------
def find_timestamp_for_text(result_text, mapping, tolerance=15):
    for entry in mapping:
        if result_text.strip() in entry["text"]:
            return entry["timestamp"]
    # fallback: fuzzy match
    for entry in mapping:
        if result_text.strip()[:tolerance] in entry["text"]:
            return entry["timestamp"]
    return None

# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(page_title="Video QA", layout="wide")
st.title("🎥 Chat with the Video")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Select retriever
retriever_type = st.selectbox("Choose Retrieval Method:", RETRIEVER_OPTIONS)

# Input box for user question
query = st.chat_input("Ask a question about the video...")

# Instantiate retriever
retriever = UnifiedRetriever(
    use_faiss="faiss" in retriever_type,
    use_pgvector="pgvector" in retriever_type,
    pg_method="ivfflat",  # or "hnsw"
    use_multimodal="multimodal" in retriever_type
)

# Load mapping
mapping = load_mapping(MAPPING_PATH)

# Display chat history
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

# Process new query
if query:
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Perform retrieval
    results = retriever.search(query, retriever_type=retriever_type, top_k=1)

    # Show top result or fallback message
    with st.chat_message("assistant"):
        if not results:
            st.markdown("🤖 Sorry, I couldn't find anything relevant in the video.")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "🤖 Sorry, I couldn't find anything relevant in the video."
            })
        else:
            result_text, score_or_distance = results[0]
            timestamp = find_timestamp_for_text(result_text, mapping)

            if timestamp is not None:
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                timestamp_str = f"{minutes:02d}:{seconds:02d}"

                st.markdown(f"**📍 Found at {timestamp_str}**")
                st.markdown(f"**💬 Answer:** {result_text}")
                st.video(VIDEO_PATH, start_time=int(timestamp))

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"📍 Found at {timestamp_str}\n\n💬 {result_text}"
                })
            else:
                st.markdown("🤖 I found an answer, but couldn't locate the exact timestamp.")
                st.markdown(f"💬 {result_text}")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"💬 {result_text} (timestamp unavailable)"
                })
