from typing import Dict, List

import chromadb
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "engineering_manuals"


class VectorStore:
    """Store and retrieve engineering document chunks using ChromaDB."""

    def __init__(
        self,
        model: SentenceTransformer,
        persist_directory: str = "data/chroma",
    ):
        self.model = model
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Dict], batch_size: int = 64) -> None:
        """Embed and persist chunks with source metadata."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            valid_chunks = [
                chunk for chunk in batch
                if chunk.get("text", "").strip()
            ]
            if not valid_chunks:
                continue

            texts = [chunk["text"] for chunk in valid_chunks]
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).tolist()

            self.collection.upsert(
                ids=[chunk["chunk_id"] for chunk in valid_chunks],
                documents=texts,
                embeddings=embeddings,
                metadatas=[
                    {
                        "document_name": chunk["document_name"],
                        "page_number": int(chunk["page_number"]),
                    }
                    for chunk in valid_chunks
                ],
            )

    
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        max_distance: float | None = None,
    ) -> List[Dict]:
        """Retrieve relevant chunks, optionally filtering weak matches.

        max_distance is a cosine-distance cutoff. Lower distances are
        more similar. The cutoff must be calibrated on evaluation data.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if max_distance is not None and not 0 <= max_distance <= 2:
            raise ValueError("max_distance must be between 0 and 2.")

        collection_count = self.collection.count()
        if collection_count == 0:
            return []

        # Fetch extra candidates when filtering is enabled. Otherwise,
        # weak matches among the first top_k could hide better candidates.
        candidate_count = top_k
        if max_distance is not None:
            candidate_count = top_k * 3

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(candidate_count, collection_count),
            include=["documents", "metadatas", "distances"],
        )

        matches = []
        for i, text in enumerate(results["documents"][0]):
            distance = float(results["distances"][0][i])

            if max_distance is not None and distance > max_distance:
                continue

            matches.append({
                "text": text,
                "document_name": results["metadatas"][0][i]["document_name"],
                "page_number": results["metadatas"][0][i]["page_number"],
                "distance": distance,
            })

            if len(matches) == top_k:
                break

        return matches

        
    def reset(self) -> None:
        """Clear the collection before rebuilding the index."""
        self.client.delete_collection(name=COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        
