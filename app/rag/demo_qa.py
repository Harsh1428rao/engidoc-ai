
# from app.retrieval import vector_store
from app.retrieval.embeddings import load_embedding_model
from app.retrieval.vector_store import VectorStore
from app.rag.generator import RAGGenerator
from app.rag.qa import EngineeringQA
from app.retrieval.reranker import PassageReranker

def main():
    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    print("Connecting to the persistent vector store...")
    store = VectorStore(
        model=embedding_model,
        persist_directory="data/chroma",
    )

    chunk_count = store.collection.count()

    if chunk_count == 0:
        print("The vector store is empty.")
        print(
            "Run the retrieval indexing demo first to index "
            "sample_docs/engineering_manual.pdf."
        )
        return

    print(f"Loaded {chunk_count} indexed chunks.")

    try:
        generator = RAGGenerator()
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return

    reranker = PassageReranker()

    qa = EngineeringQA(
        vector_store=store,
        generator=generator,
        top_k=5,
        max_distance=0.65,
        reranker=reranker,
    )

    print("\nEngiDoc AI is ready.")
    print("Ask a question about the engineering manual.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            question = input("\nYour question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting EngiDoc AI.")
            break

        if question.lower() in {"exit", "quit"}:
            print("Exiting EngiDoc AI.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            result = qa.ask(question)
        except Exception as exc:
            print(f"\nError while answering the question: {exc}")
            continue

        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(result["answer"])

        print("\n" + "-" * 70)
        print("SOURCE MAPPING")
        print("-" * 70)

        if result["sources"]:
            for source in result["sources"]:
                print(
                    f"[{source['source_id']}] "
                    f"{source['document_name']} | "
                    f"page {source['page_number']} | "
                    f"cosine distance {source['distance']:.4f}"
                )
        else:
            print("No source passages were retrieved.")

        print("\n" + "-" * 70)
        print("RETRIEVED PASSAGES (DEBUG)")
        print("-" * 70)

        if result["contexts"]:
            for index, context in enumerate(
                result["contexts"],
                start=1,
            ):
                print(
                    f"\n[S{index}] "
                    f"{context['document_name']} | "
                    f"page {context['page_number']} | "
                    f"cosine distance {context['distance']:.4f}"
                )
                print("-" * 50)
                print(context["text"])
        else:
            print("No relevant passages were retrieved.")

        print("\n" + "=" * 70)
        print(f"Retrieved chunks: {result['retrieved_chunks']}")
        print("=" * 70)


if __name__ == "__main__":
    main()
