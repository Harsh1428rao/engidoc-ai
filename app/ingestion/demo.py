
from pprint import pprint

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages


def main():
    pdf_path = input("Enter the path to an engineering PDF: ").strip()

    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages, chunk_size=1000, overlap=150)

    total_characters = sum(len(page["text"]) for page in pages)

    print("\n--- Ingestion Report ---")
    print(f"Pages processed: {len(pages)}")
    print(f"Characters extracted: {total_characters}")
    print(f"Chunks created: {len(chunks)}")

    if not chunks:
        print(
            "No extractable text found. "
            "The PDF may be scanned, empty, or image-only."
        )
        return

    print("\n--- First Chunk ---")
    pprint(chunks[0])


if __name__ == "__main__":
    main()



