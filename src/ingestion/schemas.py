from pydantic import BaseModel
from typing import List


class TextChunk(BaseModel):
    chunk_id: int
    text: str
    page_number: int


class IngestionResult(BaseModel):
    document_name: str
    total_pages: int
    total_chunks: int
    chunks: List[TextChunk]
