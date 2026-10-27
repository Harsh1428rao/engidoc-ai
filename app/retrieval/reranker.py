
from typing import Dict, List

from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class PassageReranker:
    """Rerank retrieved passages by their relevance to a question."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        question: str,
        passages: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not passages:
            return []

        pairs = [
            (question.strip(), passage["text"])
            for passage in passages
        ]
        scores = self.model.predict(pairs)

        ranked = []
        for passage, score in zip(passages, scores):
            result = dict(passage)
            result["rerank_score"] = float(score)
            ranked.append(result)

        ranked.sort(
            key=lambda passage: passage["rerank_score"],
            reverse=True,
        )
        return ranked[:top_k]
