

from fastapi import FastAPI

app = FastAPI(title="Local RAG Chatbot", version="0.1.0")


@app.get("/health")
def health_check() -> dict:
    # Simple health endpoint for verifying the service is running.
    return {"status": "ok"}


from pathlib import Path
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_text
from src.ingestion.schemas import IngestionResult

DATA_DIR = Path("data/raw")


@app.post("/ingest/pdf", response_model=IngestionResult)
def ingest_pdf(filename: str) -> IngestionResult:
    pdf_path = DATA_DIR / filename

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {filename}")

    total_pages, pages = load_pdf(pdf_path)
    chunks = chunk_text(pages)

    return IngestionResult(
        document_name=filename,
        total_pages=total_pages,
        total_chunks=len(chunks),
        chunks=chunks,
    )

from pydantic import BaseModel
from src.embeddings.st_embedder import SentenceTransformerEmbedder
from src.vector_store.simple_store import SimpleVectorStore

class IndexPdfRequest(BaseModel):
    filename: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

store = SimpleVectorStore(persist_path="data/vectors/store.npz")
embedder = SentenceTransformerEmbedder(model_name="intfloat/e5-small-v2")

@app.post("/index/pdf")
def index_pdf(payload: IndexPdfRequest) -> dict:
    pdf_path = DATA_DIR / payload.filename
    if not pdf_path.exists():
        # Keep it simple for now; we can switch to HTTPException later.
        raise FileNotFoundError(f"PDF not found: {payload.filename}")

    total_pages, pages = load_pdf(pdf_path)
    chunks = chunk_text(pages)

    texts = [c.text for c in chunks]
    metadatas = [
        {"document": payload.filename, "page_number": c.page_number, "chunk_id": c.chunk_id}
        for c in chunks
    ]

    embeddings = embedder.embed(texts)
    store.add(embeddings=embeddings, texts=texts, metadatas=metadatas)

    return {"document_name": payload.filename, "total_pages": total_pages, "indexed_chunks": len(chunks)}


@app.post("/search")
def search(payload: SearchRequest) -> dict:
    query_vec = embedder.embed([payload.query])[0]
    hits = store.search(query_embedding=query_vec, top_k=payload.top_k)

    return {
        "query": payload.query,
        "top_k": payload.top_k,
        "hits": [
            {"score": h.score, "text": h.text, "metadata": h.metadata}
            for h in hits
        ],
    }
