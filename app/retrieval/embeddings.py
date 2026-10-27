from typing import List

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
) -> SentenceTransformer:
    """Load a local sentence-transformer embedding model."""
    return SentenceTransformer(model_name)


def embed_texts(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> List[List[float]]:
    """Convert text passages into normalized embedding vectors."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if not texts:
        return []

    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("All texts must be non-empty strings.")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > batch_size,
    )

    return embeddings.tolist()
