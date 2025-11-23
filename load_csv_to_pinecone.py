import os, uuid
import pandas as pd
from tqdm import tqdm
from pinecone import Pinecone
from openai import OpenAI

# --- env variables from Codespaces Secrets ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
PINECONE_HOST = os.getenv("PINECONE_HOST", "").strip()
INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-chat-demo").strip()

CSV_PATH = "kids_clothing_catalog_5000.csv"

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY missing")
if not PINECONE_HOST:
    raise ValueError("PINECONE_HOST missing")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def embed_batch(texts):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in res.data]

def chunk_text(text, chunk_size=900, overlap=150):
    text = str(text).replace("\n", " ")
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def read_csv_rows(path):
    df = pd.read_csv(path)
    cols = list(df.columns)

    rows = []
    for i, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in cols]
        row_text = " | ".join(parts)
        rows.append((i + 1, row_text))
    return rows, cols

def upsert_csv(batch_size=64):
    rows, cols = read_csv_rows(CSV_PATH)
    print(f"📦 CSV уншсан мөр: {len(rows)}")
    print("⏳ Embedding → Pinecone-д илгээж эхэлж байна...")

    vectors_batch = []
    texts_batch = []
    meta_batch = []

    for row_no, text in tqdm(rows, desc="Preparing rows"):
        chunks = chunk_text(text)
        for c in chunks:
            texts_batch.append(c)
            meta_batch.append({
                "text": c,
                "source": CSV_PATH,
                "row": row_no,
                "columns": cols
            })

            if len(texts_batch) >= batch_size:
                embs = embed_batch(texts_batch)
                for emb, meta in zip(embs, meta_batch):
                    vectors_batch.append((str(uuid.uuid4()), emb, meta))
                index.upsert(vectors=vectors_batch)

                vectors_batch, texts_batch, meta_batch = [], [], []

    # flush last
    if texts_batch:
        embs = embed_batch(texts_batch)
        for emb, meta in zip(embs, meta_batch):
            vectors_batch.append((str(uuid.uuid4()), emb, meta))
        index.upsert(vectors=vectors_batch)

    print(f"✅ CSV Pinecone-д орлоо. Нийт chunks: {len(rows)}")

if __name__ == "__main__":
    upsert_csv()
