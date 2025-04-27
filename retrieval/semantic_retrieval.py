import faiss
import numpy as np
import pickle
import psycopg2
from sqlalchemy import create_engine
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, MetaData, Table, select
from tqdm import tqdm

# ------------------------
# CONFIG
# ------------------------
TEXT_EMBEDDINGS_PATH = "embeddings/text/text_embeddings.pkl"
IMAGE_EMBEDDINGS_PATH = "embeddings/images/image_embeddings.pkl"

# PostgreSQL Database URL
# Example: "postgresql://username:password@localhost:5432/rag_db"
DATABASE_URL = "postgresql://postgres:password@localhost:5432/rag_db"

# ------------------------
# FAISS Retriever
# ------------------------

class FaissRetriever:
    def __init__(self, embeddings_path=TEXT_EMBEDDINGS_PATH):
        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)
        self.texts = data['texts']
        self.embeddings = np.array(data['embeddings']).astype('float32')

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

        print(f"[INFO] FAISS index built with {self.index.ntotal} vectors.")

    def search(self, query_embedding, top_k=5):
        distances, indices = self.index.search(query_embedding, top_k)
        results = [(self.texts[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
        return results

# ------------------------
# PostgreSQL Retriever (pgvector)
# ------------------------

class PgvectorRetriever:
    def __init__(self, database_url=DATABASE_URL, table_name="text_embeddings"):
        self.engine = create_engine(database_url)
        self.table_name = table_name
        self.metadata = MetaData()
        self.table = Table(
            self.table_name, self.metadata,
            Column('id', Integer, primary_key=True),
            Column('text', String),
            Column('embedding', Vector(384))  # adjust 384 if using other embedding size
        )
        self.metadata.create_all(self.engine)

    def load_embeddings(self, embeddings_path=TEXT_EMBEDDINGS_PATH):
        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)
        self.texts = data['texts']
        self.embeddings = np.array(data['embeddings'])

    def populate(self):
        self.load_embeddings()
        with self.engine.connect() as conn:
            print("[INFO] Populating database...")
            conn.execute(self.table.delete())  # clear old
            for idx, (text, embed) in tqdm(enumerate(zip(self.texts, self.embeddings))):
                conn.execute(self.table.insert().values(
                    id=idx,
                    text=text,
                    embedding=embed.tolist()
                ))

    def search(self, query_embedding, top_k=5, method="ivfflat"):
        with self.engine.connect() as conn:
            if method.lower() == "ivfflat":
                stmt = select(self.table).order_by(self.table.c.embedding.l2_distance(query_embedding)).limit(top_k)
            elif method.lower() == "hnsw":
                stmt = select(self.table).order_by(self.table.c.embedding.max_inner_product(query_embedding)).limit(top_k)
            else:
                raise ValueError("Unknown method: choose 'ivfflat' or 'hnsw'.")

            results = conn.execute(stmt).fetchall()

        return [(row['text'], None) for row in results]  # no distances returned

