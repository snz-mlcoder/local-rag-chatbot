from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader


def load_pdf(file_path: Path) -> Tuple[int, List[Tuple[int, str]]]:
    """
    Load a text-based PDF and extract text per page.

    Returns:
        total_pages: int
        pages: List of (page_number, page_text)
    """
    reader = PdfReader(str(file_path))
    pages_text: List[Tuple[int, str]] = []

    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        cleaned = text.strip()
        if cleaned:
            pages_text.append((idx + 1, cleaned))

    return len(reader.pages), pages_text
