# LLM-ast5-videoRAG
This is a repository containing all the deliverables for assignment 5 of CMPS396A: LLM and RAGs course. It implemnets a multimodal RAG system around a video, combining speech-to-text transcription, keyframe extraction, text and image embeddings, semantic + lexical search, and then wrap it all inside a Streamlit chat-like interface that either plays relevant video segments or says no answer found.

# Directory Structure:
RAG-Video-QA/
├── interface/            # Streamlit app
├── data/                 # Video, Chunks, Frames and Mappings of Text to Frames
├── embeddings/           # Text & image embeddings
├── retrieval/            # Semantic and lexical search
├── models/               # Whisper, CLIP, embedding models
├── utils/                # Helper functions for video download and data extractions
├── gold test data/       # Test data in json, Evaluation of the different retreival methods
└── requirements.txt      # Dependencies


# Running the App
To run the app, open the terminal from the root directory, 
and write the command: "streamlit run interface/app.py"

To run the "gold test data/evaluate.py", first run the streamlit app, then run the evaluate.py

# Installing all the required libraries:
Command (Windows): pip install -r requirements.txt

# Tech Stack:
Programming Language: Python

Frameworks/Libraries:

- Streamlit (UI)

- OpenAI Whisper (speech-to-text)

- Hugging Face Transformers (text embeddings)

- CLIP (image embeddings)

- FAISS, pgvector (semantic search)

- Rank-BM25 (TF-IDF, BM25)

- ffmpeg-python (for frame extraction)

- yt_dlp (download YouTube video)

- psycopg2, SQLAlchemy (PostgreSQL with pgvector)

- PostgreSQL: https://www.postgresql.org/download/windows/

- pgvector extension for PostgreSQL https://github.com/pgvector/pgvector/


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


🛠 Assignment Mandatory Components
Speech-to-Text: Use Whisper.

Text Embeddings: Open-source MTEB model.

Image Embeddings: OpenAI CLIP.

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

    - Semantic search (FAISS, PostgreSQL pgvector with IVFFLAT and HNSW). (retreival/semantic_retrieval.py)

    - Lexical search (TF-IDF and BM25). (retreival/lexical_retrieval.py)
    
    - Unified Retrieval (retrieval/unified_retriever.py)

    - optional: Combine text and image retrieval. (retrieval/unified_retriever.py)

Step 6: Streamlit App: (interface/app.py)

    - Text input for question.

    - Setting sidebar to select between the different 

    - Display top matching video segments (with embedded video).

    - If no good answer, show "No answer found."

Step 7: Evaluation: 
    - Prepare Gold Test Data (gold test data/data.json)

    - Evaluate and Compare the Different Retrieval Methods (gold test data/evaluate.py)


# For postgres connection:

Make sure that you setup PosteGres, and store the url in a .env file in the following format:
    DATABASE_URL=postgresql://username:password@localhost:port_number/database_name

On Command Prompt (Windows):
If you have psql installed, run this in your system terminal, to connect to the database:
    psql -d <database_name> -U <username> -p <port_number>

Make sure to install pgvector extension for PostgreSQL following the instructions here:
https://github.com/pgvector/pgvector/

To enable the vector extension for pgvector retrieval method, run this inside your psql session:
    CREATE EXTENSION vector;
    SELECT * FROM pg_extension WHERE extname = 'vector';