
import pytest

from app.ingestion.chunker import chunk_pages


def test_chunker_preserves_page_metadata():
    pages = [
        {
            "document_name": "manual.pdf",
            "page_number": 3,
            "text": "A" * 25,
        }
    ]

    chunks = chunk_pages(pages, chunk_size=10, overlap=2)

    assert len(chunks) == 4
    assert all(c["document_name"] == "manual.pdf" for c in chunks)
    assert all(c["page_number"] == 3 for c in chunks)
    assert chunks[0]["text"] == "A" * 10
    assert chunks[1]["text"] == "A" * 10
    assert chunks[0]["chunk_id"] != chunks[1]["chunk_id"]


def test_chunks_overlap():
    pages = [
        {
            "document_name": "manual.pdf",
            "page_number": 1,
            "text": "abcdefghijklmnop",
        }
    ]

    chunks = chunk_pages(pages, chunk_size=10, overlap=3)

    assert chunks[0]["text"][-3:] == chunks[1]["text"][:3]


def test_empty_pages_are_skipped():
    pages = [
        {
            "document_name": "manual.pdf",
            "page_number": 1,
            "text": "   ",
        }
    ]

    assert chunk_pages(pages) == []


@pytest.mark.parametrize(
    "chunk_size,overlap",
    [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 12)],
)
def test_invalid_chunk_parameters_raise(chunk_size, overlap):
    pages = [
        {
            "document_name": "manual.pdf",
            "page_number": 1,
            "text": "abc",
        }
    ]

    with pytest.raises(ValueError):
        chunk_pages(
            pages,
            chunk_size=chunk_size,
            overlap=overlap,
        )
