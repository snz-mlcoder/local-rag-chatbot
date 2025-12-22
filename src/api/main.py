

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
