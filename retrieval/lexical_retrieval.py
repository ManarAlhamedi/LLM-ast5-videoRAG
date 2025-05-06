import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

# ------------------------
# CONFIG
# ------------------------
CHUNKS_DIR = "data/transcripts/chunks"

# ------------------------
# TF-IDF Retriever
# ------------------------

class TfidfRetriever:
    def __init__(self):
        self.texts = []
        self.vectorizer = TfidfVectorizer()

    def load_texts(self):
        chunk_files = sorted(Path(CHUNKS_DIR).glob("chunk_*.json"))
        self.texts = []
        for chunk_file in chunk_files:
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunk = json.load(f)
                self.texts.append(chunk["text"])

    def fit(self):
        self.load_texts()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
        print(f"[INFO] TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    def search(self, query, top_k=5, threshold=0.1):
        query_vec = self.vectorizer.transform([query])
        scores = (self.tfidf_matrix @ query_vec.T).toarray().ravel()
        top_indices = scores.argsort()[::-1][:top_k]
        if scores[top_indices[0]] < threshold:
            return [("No answer found with sufficient confidence.", 0.0)]
        return [(self.texts[idx], scores[idx]) for idx in top_indices]

# ------------------------
# BM25 Retriever
# ------------------------

class BM25Retriever:
    def __init__(self):
        self.texts = []
        self.corpus = []

    def load_texts(self):
        chunk_files = sorted(Path(CHUNKS_DIR).glob("chunk_*.json"))
        self.texts = []
        self.corpus = []
        for chunk_file in chunk_files:
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunk = json.load(f)
                text = chunk["text"]
                self.texts.append(text)
                self.corpus.append(word_tokenize(text.lower()))

    def fit(self):
        self.load_texts()
        self.bm25 = BM25Okapi(self.corpus)
        print(f"[INFO] BM25 corpus size: {len(self.corpus)}")

    def search(self, query, top_k=5, threshold=0.1):
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        if scores[top_indices[0]] < threshold:
            return [("No answer found with sufficient confidence.", 0.0)]
        return [(self.texts[idx], scores[idx]) for idx in top_indices]

