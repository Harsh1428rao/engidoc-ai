
from unittest.mock import MagicMock, patch

import pytest

from app.retrieval.reranker import PassageReranker


@pytest.fixture
def reranker():
    with patch("app.retrieval.reranker.CrossEncoder"):
        instance = PassageReranker()
        instance.model = MagicMock()
        yield instance


def test_reranks_passages_by_score(reranker):
    reranker.model.predict.return_value = [0.2, 0.9, 0.5]

    passages = [
        {"text": "Unrelated contents", "page_number": 5},
        {"text": "Voltage is shown on the rating plate", "page_number": 13},
        {"text": "General motor information", "page_number": 1},
    ]

    result = reranker.rerank("Where is the voltage shown?", passages)

    assert [item["page_number"] for item in result] == [13, 1, 5]
    assert result[0]["rerank_score"] == pytest.approx(0.9)


def test_returns_only_top_k(reranker):
    reranker.model.predict.return_value = [0.2, 0.9, 0.5]

    passages = [
        {"text": "A"},
        {"text": "B"},
        {"text": "C"},
    ]

    result = reranker.rerank("Question", passages, top_k=2)

    assert len(result) == 2
    assert [item["text"] for item in result] == ["B", "C"]


def test_empty_passages_return_empty_list(reranker):
    assert reranker.rerank("Question", []) == []
    reranker.model.predict.assert_not_called()


def test_rejects_empty_question(reranker):
    with pytest.raises(ValueError, match="Question"):
        reranker.rerank(" ", [{"text": "Evidence"}])


def test_rejects_invalid_top_k(reranker):
    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("Question", [{"text": "Evidence"}], top_k=0)
