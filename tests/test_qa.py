
from unittest.mock import MagicMock

import pytest

from app.rag.qa import EngineeringQA
# from tests.test_rag_generator import generator


@pytest.fixture
def qa():
    vector_store = MagicMock()
    generator = MagicMock()

    vector_store.search.return_value = [
        {
            "text": "The rated voltage is shown on the rating plate.",
            "document_name": "motor.pdf",
            "page_number": 13,
            "distance": 0.2,
        },
        {
            "text": "Follow the terminal-box connection diagram.",
            "document_name": "motor.pdf",
            "page_number": 13,
            "distance": 0.3,
        },
        {
            "text": "Inspect mounting alignment.",
            "document_name": "motor.pdf",
            "page_number": 11,
            "distance": 0.4,
        },
    ]

    generator.answer.return_value = (
        "Check the rating plate [motor.pdf, p. 13]."
    )

    return EngineeringQA(vector_store, generator), vector_store, generator


def test_ask_connects_retrieval_and_generation(qa):
    service, vector_store, generator = qa

    result = service.ask("How do I select the voltage?")

    vector_store.search.assert_called_once_with(
        "How do I select the voltage?",
        top_k=5,
        max_distance=0.70,
    )
    generator.answer.assert_called_once()

    assert result["retrieved_chunks"] == 3
    assert result["answer"] == (
        "Check the rating plate [motor.pdf, p. 13]."
    )


def test_sources_map_to_retrieved_chunks(qa):
    service, _, generator = qa

    generator.answer.return_value = (
        "The manual explains voltage selection [S1]. "
        "Check the rating plate [S2]. See the relevant section [S3]."
    )

    result = service.ask("How do I select the voltage?")
    

    expected_sources = [
        {
            "source_id": f"S{index}",
            "document_name": context["document_name"],
            "page_number": context["page_number"],
            "distance": context["distance"],
        }
        for index, context in enumerate(result["contexts"], start=1)
    ]

    assert result["sources"] == expected_sources
    assert len(result["sources"]) == len(result["contexts"])


@pytest.mark.parametrize("question", ["", "   "])
def test_ask_rejects_empty_question(qa, question):
    service, vector_store, generator = qa

    with pytest.raises(ValueError, match="Question"):
        service.ask(question)

    vector_store.search.assert_not_called()
    generator.answer.assert_not_called()


def test_custom_top_k_is_used(qa):
    vector_store, generator = MagicMock(), MagicMock()
    service = EngineeringQA(vector_store, generator, top_k=3)

    service.ask("What does the manual say?")

    vector_store.search.assert_called_once_with(
        "What does the manual say?",
        top_k=3,
        max_distance=0.70,
    )
