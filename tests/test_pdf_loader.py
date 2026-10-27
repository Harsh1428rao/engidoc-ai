
import pytest

from app.ingestion import pdf_loader


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class FakeReader:
    def __init__(self, path):
        self.is_encrypted = False
        self.pages = [
            FakePage("  Motor operating instructions  "),
            FakePage("Safety precautions"),
            FakePage(None),
        ]


def test_load_pdf_preserves_page_metadata(tmp_path, monkeypatch):
    pdf_path = tmp_path / "manual.pdf"
    pdf_path.touch()

    monkeypatch.setattr(pdf_loader, "PdfReader", FakeReader)

    pages = pdf_loader.load_pdf(pdf_path)

    assert len(pages) == 3
    assert pages[0] == {
        "document_name": "manual.pdf",
        "page_number": 1,
        "text": "Motor operating instructions",
    }
    assert pages[1]["page_number"] == 2
    assert pages[2]["text"] == ""


def test_missing_pdf_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        pdf_loader.load_pdf(tmp_path / "missing.pdf")


def test_non_pdf_file_raises_error(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.touch()

    with pytest.raises(ValueError, match="Only PDF"):
        pdf_loader.load_pdf(file_path)


def test_encrypted_pdf_raises_error(tmp_path, monkeypatch):
    pdf_path = tmp_path / "encrypted.pdf"
    pdf_path.touch()

    class EncryptedReader:
        is_encrypted = True

        def __init__(self, path):
            self.pages = []

    monkeypatch.setattr(pdf_loader, "PdfReader", EncryptedReader)

    with pytest.raises(ValueError, match="Encrypted"):
        pdf_loader.load_pdf(pdf_path)
