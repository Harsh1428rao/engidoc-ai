
from unittest.mock import MagicMock

import pytest

from app.retrieval.vector_store import VectorStore


@pytest.fixture
def store():
    vector_store = VectorStore.__new__(VectorStore)
    vector_store.model = MagicMock()
    vector_store.client = MagicMock()
    vector_store.collection = MagicMock()
    vector_store.model.encode.return_value.tolist.return_value = [
        [0.1, 0.2, 0.3]
    ]
    return vector_store


def test_search_returns_all_results(store):
    store.collection.count.return_value = 3
    store.collection.query.return_value = {
        "documents": [["alignment instructions", "rating plate", "cable sizing"]],
        "metadatas": [[
            {"document_name": "manual.pdf", "page_number": 11},
            {"document_name": "manual.pdf", "page_number": 13},
            {"document_name": "manual.pdf", "page_number": 13},
        ]],
        "distances": [[0.1, 0.2, 0.3]],
    }

    results = store.search("motor installation", top_k=3)

    assert len(results) == 3
    assert results[0]["text"] == "alignment instructions"
    assert results[1]["page_number"] == 13
    assert results[2]["distance"] == pytest.approx(0.3)


def test_search_returns_empty_for_empty_collection(store):
    store.collection.count.return_value = 0

    assert store.search("motor installation") == []
    store.collection.query.assert_not_called()


@pytest.mark.parametrize("query", ["", "   "])
def test_search_rejects_empty_query(store, query):
    with pytest.raises(ValueError):
        store.search(query)


def test_search_rejects_invalid_top_k(store):
    with pytest.raises(ValueError):
        store.search("motor installation", top_k=0)
