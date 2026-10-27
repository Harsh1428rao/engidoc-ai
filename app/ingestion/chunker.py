
from typing import Dict, List


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = 1000,
    overlap: int = 150,
    min_chunk_chars: int = 1,
) -> List[Dict]:
    """Split page text into overlapping chunks with source metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    if min_chunk_chars < 1:
        raise ValueError("min_chunk_chars must be positive.")

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        text = page["text"].strip()

        if not text:
            continue

        for start in range(0, len(text), step):
            chunk_text = text[start : start + chunk_size].strip()

            if len(chunk_text) < min_chunk_chars:
                continue

            chunks.append({
                "chunk_id": (
                    f"{page['document_name']}:"
                    f"{page['page_number']}:{start}"
                ),
                "document_name": page["document_name"],
                "page_number": page["page_number"],
                "text": chunk_text,
            })

    return chunks
