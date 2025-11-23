import os, uuid
from tqdm import tqdm
from pypdf import PdfReader
from pinecone import Pinecone
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
PINECONE_HOST = os.getenv("PINECONE_HOST", "").strip()
INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-chat-demo").strip()

PDF_PATH = "fivebaby_pitch.pdf"

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY missing")
if not PINECONE_HOST:
    raise ValueError("PINECONE_HOST missing")
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(name=INDEX_NAME, host=PINECONE_HOST)

def embed_text(text):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding

def chunk_text(text, chunk_size=900, overlap=150):
    text = text.replace("\n", " ")
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def read_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text()
            if txt and txt.strip():
                pages.append((i + 1, txt))
        except Exception:
            continue
    return pages

def upsert_pdf(batch_size=50):
    pages = read_pdf(PDF_PATH)
    print(f"📄 PDF-тэй нийт {len(pages)} хуудас байна.")

    batch = []
    for page_no, text in tqdm(pages, desc="Embedding PDF"):
        for c in chunk_text(text):
            emb = embed_text(c)
            batch.append((str(uuid.uuid4()), emb, {
                "source": PDF_PATH,
                "page": page_no,
                "text": c
            }))
            if len(batch) >= batch_size:
                index.upsert(vectors=batch)
                batch = []

    if batch:
        index.upsert(vectors=batch)

    print("✅ PDF Pinecone-д амжилттай орлоо!")

if __name__ == "__main__":
    upsert_pdf()
