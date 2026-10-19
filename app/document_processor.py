
from pypdf import PdfReader
from pathlib import Path


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""

    pdf_path = Path(file_path)
    reader = PdfReader(str(pdf_path))

    extracted_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            extracted_pages.append(
                f"[Page {page_number}]\n{page_text}"
            )

    return "\n\n".join(extracted_pages)
