# LLM-ast5-videoRAG
This is a repo containing all the deliverables for assignment 5 of CMPS396A: LLM and RAGs course. It implements a system to answer user questions about a certain video.

# Tech Stack:
Programming Language: Python

Frameworks/Libraries:

- Streamlit (UI)

- OpenAI Whisper (speech-to-text)

- Hugging Face Transformers (text embeddings)

- CLIP (image embeddings)

- FAISS, pgvector (semantic search)

- Scikit-learn or Rank-BM25 (TF-IDF, BM25)

- moviepy, opencv or ffmpeg-python (for frame extraction)

- yt_dlp (download YouTube video)

- psycopg2, SQLAlchemy (PostgreSQL with pgvector)


# Directory Structure:
RAG-Video-QA/
├── app/                  # Streamlit app
├── data/                 # Video, Chunks, Frames and Mappings of Text to Frames
├── embeddings/           # Text & image embeddings
├── retrieval/            # Semantic and lexical search
├── models/               # Whisper, CLIP, embedding models
├── utils/                # Helper functions for video download and data extractions
└── requirements.txt      # Dependencies


(1) Text Embedding
Model from Hugging Face: sentence-transformers/all-MiniLM-L6-v2

✅ Lightweight (fits easily in RAM)

✅ Very good performance on MTEB tasks

✅ Super fast

(2) Image Embedding
Model: openai/clip-vit-base-patch32

✅ Free and open-source

✅ Works super well for encoding images into embeddings

✅ Many libraries already support it easily (like CLIP from Hugging Face or OpenAI’s repo).


🛠 Project Mandatory Components
Speech-to-Text: Use Whisper.

Text Embeddings: Open-source MTEB model.

Image Embeddings: OpenAI CLIP or similar.

Retrieval: FAISS + PostgreSQL+pgvector (IVFFLAT + HNSW) + TF-IDF + BM25.

Streamlit App: Search, visualize, handle "no answer", friendly UI.

Gold Test Set: 10 answerable + 5 unanswerable questions.

Evaluation: Accuracy, rejection quality, latency, graphs/tables.

Video Demo: Clear 10-min walkthrough.

(Optional Bonus): Score fusion, fine-tuning, explainability UI.