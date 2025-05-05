# LLM-ast5-videoRAG
This is a repository containing all the deliverables for assignment 5 of CMPS396A: LLM and RAGs course. It implemnets a multimodal RAG system around a video, combining speech-to-text transcription, keyframe extraction, text and image embeddings, semantic + lexical search, and then wrap it all inside a Streamlit chat-like interface that either plays relevant video segments or says no answer found.

# Rnnning the App
To run the app, open the terminal from the root directory, 
and write the command: "streamlit run interface/app.py"

# Tech Stack:
Programming Language: Python

Frameworks/Libraries:

- Streamlit (UI)

- OpenAI Whisper (speech-to-text)

- Hugging Face Transformers (text embeddings)

- CLIP (image embeddings)

- FAISS, pgvector (semantic search)

- Scikit-learn or Rank-BM25 (TF-IDF, BM25)

- ffmpeg-python (for frame extraction)

- yt_dlp (download YouTube video)

- psycopg2, SQLAlchemy (PostgreSQL with pgvector)

- PostgreSQL: https://www.postgresql.org/download/windows/

- pgvector extension for PostgreSQL https://github.com/pgvector/pgvector/

# Directory Structure:
RAG-Video-QA/
├── interface/                  # Streamlit app
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


Pipeline:

Step 1: Download video from YouTube (utils/download_YT_video.py).

Step 2: Extract audio, transcribe with Whisper, chunk text, extract keyframes, match frames to text (utils/data_extraction.py).

Step 3: Text embedding with Sentence Transformers (models/text_embedder.py).

Step 4: Image embedding with CLIP (models/image_embedder.py).

Step 5: Retrieval system:

    - Semantic search (FAISS, PostgreSQL pgvector with IVFFLAT and HNSW).

    - Lexical search (TF-IDF and BM25).

    - ptional: Combine text and image retrieval.

Step 6: Streamlit App:

    - Text input for question.

    - Display top matching video segments (with embedded video).

    - If no good answer, show "No answer found."


    App Design: 
    --------------------------------------------------------
|  📹 Video Player [selected frame/video segment]       |
|                                                      |
|  ✍️  Text Input (User Question)                       |
|                                                      |
|  🔍 Retrieval Mode:  [Semantic | Lexical]  (Dropdown) |
|  📑 Top-k Results:   [1-10] (Slider)                  |
|                                                      |
|  💬 Answers (Text + Frame Snapshots)                 |
|                                                      |
--------------------------------------------------------
Core flow:

    User types a question.

    Chooses semantic or lexical retrieval.

    App shows best matching text chunks and frames.

    Clickable thumbnails to jump to video points.