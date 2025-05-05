import numpy as np
from retrieval.lexical_retrieval import TfidfRetriever, BM25Retriever
from retrieval.semantic_retrieval import FaissRetriever, PgvectorRetriever
from sentence_transformers import SentenceTransformer
import torch
import pickle
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# ------------------------
# CONFIG
# ------------------------
TEXT_EMBEDDINGS_PATH = "embeddings/text/text_embeddings.pkl"
IMAGE_EMBEDDINGS_PATH = "embeddings/images/image_embeddings.pkl"
FRAME_MAPPING_PATH = "data/frames/frame_mapping.pkl"

class UnifiedRetriever:
    def __init__(self, use_faiss=True, use_pgvector=False, pg_method="ivfflat", use_multimodal=False):
        self.tfidf = TfidfRetriever()
        self.bm25 = BM25Retriever()

        self.use_faiss = use_faiss
        self.use_pgvector = use_pgvector
        self.pg_method = pg_method
        self.use_multimodal = use_multimodal

        if use_faiss:
            self.faiss = FaissRetriever()

        if use_pgvector:
            self.pg = PgvectorRetriever()
            if not self.pg.is_populated(): 
                self.pg.populate()
                self.pg.create_indices()

        if use_multimodal:
            self.load_image_data()
            self.init_clip()

        self.text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def load_image_data(self):
        with open(IMAGE_EMBEDDINGS_PATH, "rb") as f:
            data = pickle.load(f)
        self.frame_texts = data["texts"]
        self.image_embeddings = np.array(data["embeddings"]).astype("float32")

    def init_clip(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def search(self, query, retriever_type="bm25", top_k=5):
        if retriever_type == "tfidf":
            self.tfidf.fit()
            return self.tfidf.search(query, top_k)

        elif retriever_type == "bm25":
            self.bm25.fit()
            return self.bm25.search(query, top_k)

        elif retriever_type == "faiss":
            query_embed = self.text_encoder.encode([query], normalize_embeddings=True).astype("float32")
            return self.faiss.search(query_embed, top_k)

        elif retriever_type == "pgvector_ivfflat":
            query_embed = self.text_encoder \
                              .encode([query], normalize_embeddings=True) \
                              .tolist()[0]
            print(f"[DEBUG] Query embedding (first 5 values): {query_embed[:5]}")
            results = self.pg.search(query_embed, top_k, method="ivfflat") 
            print(f"[DEBUG] Retrieved {len(results)} pgvector results.")

            return results

        elif retriever_type == "pgvector_hnsw":
            query_embed = self.text_encoder \
                              .encode([query], normalize_embeddings=True) \
                              .tolist()[0]
            print(f"[DEBUG] Query embedding (first 5 values): {query_embed[:5]}")
            results = self.pg.search(query_embed, top_k, method="hnsw")  
            print(f"[DEBUG] Retrieved {len(results)} pgvector results.")
            return results

        elif retriever_type == "multimodal":
            return self.search_multimodal(query, top_k)

        else:
            raise ValueError("Unsupported retriever type")


# concatenation is used for fusion
    def search_multimodal(self, query, top_k=5):
        # Encode query using CLIP
        inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            query_embed = self.clip_model.get_text_features(**inputs)
            query_embed = torch.nn.functional.normalize(query_embed, dim=-1)

        # Compute cosine similarity
        image_embeds_tensor = torch.tensor(self.image_embeddings)
        scores = torch.matmul(query_embed, image_embeds_tensor.T).squeeze(0)
        top_indices = torch.topk(scores, k=top_k).indices.tolist()

        results = [(self.frame_texts[i], scores[i].item()) for i in top_indices]
        return results
