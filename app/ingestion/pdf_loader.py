from pathlib import Path
from typing import Union

from pypdf import PdfReader

def load_pdf(pdf_path: Union[str, Path]) -> list:
    """
    Extract text from a PDF while preserving 1-based page numbers.

    Returns:
        A list of dictionaries containing document name,
        page number, and extracted text.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the file is not a supported, readable PDF.
    """
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    try:
        reader = PdfReader(str(path))

        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            pages.append(
                {
                    "document_name": path.name,
                    "page_number": page_number,
                    "text": text.strip(),
                }
            )

        return pages

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read PDF: {exc}") from exc
