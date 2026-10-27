
from statistics import mean

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages


def main():
    pdf_path = "sample_docs/engineering_manual.pdf"

    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages, chunk_size=1000, overlap=150)

    lengths = [len(chunk["text"]) for chunk in chunks]

    print("\n--- Chunk Quality Report ---")
    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Average chunk length: {mean(lengths):.1f} characters")
    print(f"Shortest chunk: {min(lengths)} characters")
    print(f"Longest chunk: {max(lengths)} characters")
    print(
        "All chunks have page metadata:",
        all(
            chunk.get("page_number") and chunk.get("document_name")
            for chunk in chunks
        ),
    )

    print("\n--- Example chunks ---")
    for chunk in chunks[:3]:
        print(
            f"\nPage {chunk['page_number']} | "
            f"{len(chunk['text'])} characters"
        )
        print(chunk["text"][:200].replace("\n", " "))


if __name__ == "__main__":
    main()
