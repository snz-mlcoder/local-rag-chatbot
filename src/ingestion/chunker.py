from typing import List
from src.ingestion.schemas import TextChunk


def chunk_text(
    pages: List[tuple],
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[TextChunk]:
    """
    Split text into overlapping chunks.
    """
    chunks: List[TextChunk] = []
    chunk_id = 0

    for page_number, text in pages:
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    page_number=page_number,
                )
            )

            chunk_id += 1
            start = end - overlap

    return chunks
