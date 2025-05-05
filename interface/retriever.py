# This will unify semantic + lexical under one interface.

from retrieval.semantic_retrieval import FaissRetriever, PgvectorRetriever
from retrieval.lexical_retrieval import TfidfRetriever, BM25Retriever

class Retriever:
    def __init__(self, mode="semantic", backend="faiss"):
        self.mode = mode
        self.backend = backend

        if self.mode == "semantic":
            if self.backend == "faiss":
                self.retriever = FaissRetriever()
            elif self.backend == "pgvector":
                self.retriever = PgvectorRetriever()
            else:
                raise ValueError(f"Unknown semantic backend {backend}")
        elif self.mode == "lexical":
            if self.backend == "tfidf":
                self.retriever = TfidfRetriever()
                self.retriever.fit()
            elif self.backend == "bm25":
                self.retriever = BM25Retriever()
                self.retriever.fit()
            else:
                raise ValueError(f"Unknown lexical backend {backend}")
        else:
            raise ValueError(f"Unknown retrieval mode {mode}")

    def search(self, query, top_k=5):
        if self.mode == "semantic":
            # Assume FAISS needs embedding input
            from models.text_embedder import TextEmbedder
            embedder = TextEmbedder()
            query_emb = embedder.embed_texts([query])
            query_emb = query_emb.astype('float32')
            return self.retriever.search(query_emb, top_k=top_k)
        else:
            return self.retriever.search(query, top_k=top_k)
