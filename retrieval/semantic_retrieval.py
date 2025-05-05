import faiss
import numpy as np
import pickle
from sqlalchemy import create_engine, select, desc
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, MetaData, Table, select,  text
from tqdm import tqdm
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# ------------------------
# CONFIG
# ------------------------
TEXT_EMBEDDINGS_PATH = "embeddings/text/text_embeddings.pkl"
IMAGE_EMBEDDINGS_PATH = "embeddings/images/image_embeddings.pkl"

# PostgreSQL Database URL
# DATABASE_URL = "postgresql://postgres:password@localhost:5432/rag_db"
DATABASE_URL = os.getenv("DATABASE_URL")

# ------------------------
# FAISS Retriever
# ------------------------

class FaissRetriever:
    def __init__(self, embeddings_path=TEXT_EMBEDDINGS_PATH):
        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)
        self.texts = data['texts']
        self.embeddings = np.array(data['embeddings']).astype('float32')

        print("[DEBUG] FAISS embeddings shape:", self.embeddings.shape) 

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
            Column('embedding', Vector(384)) 
        )
        self.metadata.create_all(self.engine)

    def load_embeddings(self, embeddings_path=TEXT_EMBEDDINGS_PATH):
        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)
        self.texts = data['texts']
        self.embeddings = np.array(data['embeddings'])
           
        print("[DEBUG] Loaded embeddings shape:", self.embeddings.shape)


    def create_indices(self):
        with self.engine.connect() as conn:
            # Ensure pgvector extension is available
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
            # Create IVFFlat index
            conn.execute(text("""
                DROP INDEX IF EXISTS text_embedding_idx;
                CREATE INDEX text_embedding_idx ON text_embeddings USING ivfflat (embedding vector_l2_ops) 
                WITH (lists = 10);
                ANALYZE text_embeddings;
            """))

            #    CREATE INDEX IF NOT EXISTS text_embedding_idx
            #     ON text_embeddings USING ivfflat (embedding vector_l2_ops)
            #     WITH (lists = 100);
            
            # Create HNSW index
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS text_embedding_hnsw_idx
                ON text_embeddings USING hnsw (embedding vector_ip_ops)
                WITH (ef_construction = 200, m = 16);
            """))
            print("[INFO] Indices created successfully.")


    def populate(self):
        self.load_embeddings()

        print(f"[DEBUG] Number of texts: {len(self.texts)}")
        print(f"[DEBUG] Number of embeddings: {len(self.embeddings)}")

        with self.engine.begin() as conn: # <-- ensures commit
            print("[INFO] Populating database...")
            conn.execute(self.table.delete())  # clear old
            for idx, (textt, embed) in tqdm(enumerate(zip(self.texts, self.embeddings))):
                print(f"[DEBUG] Inserting row {idx}")

                conn.execute(self.table.insert().values(
                    id=idx,
                    text=textt,
                    embedding=embed.tolist()
                ))
            # conn.execute(text("ANALYZE text_embeddings;"))

     



    def is_populated(self):
        with self.engine.connect() as conn:
            result = conn.execute(select(self.table.c.id).limit(1)).fetchone()
            return result is not None



    def search(self, query_embedding, top_k=5, method="ivfflat"):
        with self.engine.connect() as conn:
            if method.lower() == "ivfflat":
                distance_expr = self.table.c.embedding.l2_distance(query_embedding).label("score")
                stmt = select(self.table.c.text, distance_expr).order_by(distance_expr).limit(top_k)
            elif method.lower() == "hnsw":
                # Negative of max_inner_product to return high similarity (DESC order)
                similarity_expr = (self.table.c.embedding.max_inner_product(query_embedding) * -1).label("score")
                stmt = select(self.table.c.text, similarity_expr).order_by(desc("score")).limit(top_k)
            else:
                raise ValueError("Unknown method: choose 'ivfflat' or 'hnsw'.")

            results = conn.execute(stmt).mappings().fetchall()

            print(f"[DEBUG] Retrieved {len(results)} results from pgvector using method '{method}'.")
            for row in results:
                print(f"[DEBUG] Result text: {row['text']}, Score: {row['score']}")


        return [(row['text'], row['score']) for row in results]


