import os
import torch
import numpy as np
import pickle
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ----------------------------------
# CONFIG
# ----------------------------------
CHUNKS_DIR = "data/transcripts/chunks"
TEXT_EMBEDDINGS_DIR = "embeddings/text"
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
FILENAME_PREFIX = "text_embeddings"
# ----------------------------------

class TextEmbedder:
    def __init__(self, model_name=TEXT_MODEL_NAME, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Loading model '{model_name}' on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_texts(self, texts, batch_size=32):
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    def save_embeddings(self, embeddings, texts, output_dir=TEXT_EMBEDDINGS_DIR, filename_prefix=FILENAME_PREFIX):
        os.makedirs(output_dir, exist_ok=True)

        # Save .pkl
        pkl_path = Path(output_dir) / f"{filename_prefix}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({'texts': texts, 'embeddings': embeddings}, f)
        print(f"[INFO] Saved embeddings to {pkl_path}.")

        # Save .npz
        npz_path = Path(output_dir) / f"{filename_prefix}.npz"
        np.savez(npz_path, texts=texts, embeddings=embeddings)
        print(f"[INFO] Saved embeddings to {npz_path}.")

# ----------------------------------
# MAIN
# ----------------------------------
def main():
    # Load all text chunks
    chunk_files = sorted(Path(CHUNKS_DIR).glob("chunk_*.json"))
    texts = []

    for chunk_file in chunk_files:
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunk = json.load(f)
            texts.append(chunk["text"])

    print(f"[INFO] Loaded {len(texts)} text chunks.")

    # Embed texts
    embedder = TextEmbedder()
    embeddings = embedder.embed_texts(texts)

    # Save embeddings
    embedder.save_embeddings(embeddings, texts)

if __name__ == "__main__":
    main()
