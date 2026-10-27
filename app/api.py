
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.retrieval.embeddings import load_embedding_model
from app.retrieval.vector_store import VectorStore
from app.retrieval.reranker import PassageReranker
from app.rag.generator import RAGGenerator
from app.rag.qa import EngineeringQA


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and dependencies once when the API starts."""
    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    print("Connecting to vector store...")
    store = VectorStore(
        model=embedding_model,
        persist_directory="data/chroma",
    )

    chunk_count = store.collection.count()

    if chunk_count == 0:
        raise RuntimeError(
            "Vector store is empty. Index the engineering manual first."
        )

    print("Loading reranker...")
    reranker = PassageReranker()

    print("Initializing answer generator...")
    generator = RAGGenerator()

    app.state.qa = EngineeringQA(
        vector_store=store,
        generator=generator,
        top_k=5,
        max_distance=0.65,
        reranker=reranker,
    )

    app.state.chunk_count = chunk_count

    print(f"EngiDoc AI ready: {chunk_count} indexed chunks.")

    yield


app = FastAPI(
    title="EngiDoc AI",
    description="RAG-powered assistant for engineering documentation.",
    version="1.0.0",
    lifespan=lifespan,
)


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question about the indexed engineering documents.",
    )


class SourceResponse(BaseModel):
    source_id: str
    document_name: str
    page_number: int
    distance: float
    rerank_score: Optional[float] = None


class ContextResponse(BaseModel):
    """A retrieved passage supplied to the answer generator."""
    text: str
    document_name: str
    page_number: int
    distance: float
    rerank_score: Optional[float] = None


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]
    retrieved_chunks: int
    contexts: list[ContextResponse]


@app.get("/health")
def health():
    """Check API readiness and the number of indexed chunks."""
    qa = getattr(app.state, "qa", None)

    return {
        "status": "ok" if qa is not None else "starting",
        "indexed_chunks": getattr(app.state, "chunk_count", 0),
    }


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    """Retrieve evidence and generate a grounded answer."""
    qa = getattr(app.state, "qa", None)

    if qa is None:
        raise HTTPException(
            status_code=503,
            detail="EngiDoc AI is not ready yet.",
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=422,
            detail="Question must not be empty.",
        )

    try:
        result = qa.ask(request.question)

    except Exception as exc:
        # Keep internal exception details in the server terminal.
        print(f"Question processing failed: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Failed to process the question. Check server logs.",
        ) from exc

    return {
        "question": result["question"],
        "answer": result["answer"],
        "sources": result["sources"],
        "retrieved_chunks": result["retrieved_chunks"],
        "contexts": result["contexts"],
    }
