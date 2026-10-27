
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.rag.generator import RAGGenerator


@pytest.fixture
def generator():
    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
        with patch("app.rag.generator.Groq") as groq_class:
            instance = RAGGenerator()
            instance.client = groq_class.return_value
            yield instance


def test_prompt_contains_question_and_source():
    contexts = [
        {
            "document_name": "motor.pdf",
            "page_number": 13,
            "text": "Rated voltage is shown on the rating plate.",
        }
    ]

    prompt = RAGGenerator.build_prompt(
        "How do I select the voltage?",
        contexts,
    )

    assert "How do I select the voltage?" in prompt
    assert "[S1]" in prompt
    assert "motor.pdf" in prompt
    assert "page: 13" in prompt
    assert "Rated voltage" in prompt


def test_empty_context_returns_fallback_without_api_call(generator):
    result = generator.answer("Select the voltage", [])

    assert "can't answer reliably" in result
    generator.client.chat.completions.create.assert_not_called()


def test_answer_calls_groq_and_returns_text(generator):
    generator.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Check the rating plate [S1]."
                )
            )
        ]
    )

    contexts = [
        {
            "document_name": "motor.pdf",
            "page_number": 13,
            "text": "Rated voltage is shown on the rating plate.",
        }
    ]

    prompt = RAGGenerator.build_prompt(
        "How do I select the voltage?",
        contexts,
    )

    assert "How do I select the voltage?" in prompt
    assert "[S1]" in prompt
    assert "motor.pdf" in prompt
    assert "page: 13" in prompt
    assert "Rated voltage" in prompt

    result = generator.answer(
        "How do I select the voltage?",
        contexts,
    )

    assert result == "Check the rating plate [S1]."
    generator.client.chat.completions.create.assert_called_once()


def test_missing_api_key_raises_error():
    with patch.dict("os.environ", {}, clear=True):
        with patch("app.rag.generator.load_dotenv"):
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                RAGGenerator()


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="Question"):
        RAGGenerator.build_prompt(
            " ",
            [
                {
                    "document_name": "motor.pdf",
                    "page_number": 13,
                    "text": "Example evidence.",
                }
            ],
        )
