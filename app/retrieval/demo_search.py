from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf_loader import load_pdf
from app.retrieval.embeddings import load_embedding_model
from app.retrieval.vector_store import VectorStore


def main():
    pages = load_pdf("sample_docs/engineering_manual.pdf")

    chunks = chunk_pages(
        pages,
        chunk_size=1000,
        overlap=150,
        min_chunk_chars=80,
    )

    print("Pages loaded:", len(pages))
    print("Chunks created:", len(chunks))

    model = load_embedding_model()
    store = VectorStore(model)

    print("Resetting old index...")
    store.reset()

    print("Indexing document chunks...")
    store.add_chunks(chunks)

    print("Expected chunks:", len(chunks))
    print("Indexed chunks:", store.collection.count())

    assert store.collection.count() == len(chunks), (
        "Index count does not match the number of chunks!"
    )

    # print("Indexing document chunks...")
    # store.add_chunks(chunks)
    # print("Indexed chunks:", store.collection.count())

    while True:
        query = input("\nAsk about the motor manual (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        results = store.search(query, top_k=5)
        for rank, result in enumerate(results, start=1):
            print(
                "\n{}. {} | Page {} | Cosine distance: {:.4f}".format(
                    rank,
                    result["document_name"],
                    result["page_number"],
                    result["distance"],
                )
            )
            print(result["text"][:500])

    


if __name__ == "__main__":
    main()
