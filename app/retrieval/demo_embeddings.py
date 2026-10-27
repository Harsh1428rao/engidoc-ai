from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf_loader import load_pdf
from app.retrieval.embeddings import embed_texts, load_embedding_model


def main():
    pdf_path = "sample_docs/engineering_manual.pdf"

    print("Loading engineering manual...")
    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages, chunk_size=1000, overlap=150)
    print("Chunks created:", len(chunks))

    print("\nLoading embedding model...")
    model = load_embedding_model()

    sample_chunks = chunks[:5]
    texts = [chunk["text"] for chunk in sample_chunks]

    print("\nGenerating embeddings...")
    embeddings = embed_texts(texts, model)

    print("Embeddings generated:", len(embeddings))
    print("Embedding dimensions:", len(embeddings[0]))
    print("First vector values:", embeddings[0][:5])

    assert len(embeddings) == len(sample_chunks)
    assert all(len(vector) == 384 for vector in embeddings)

    print("\nEmbedding smoke test passed!")


if __name__ == "__main__":
    main()
