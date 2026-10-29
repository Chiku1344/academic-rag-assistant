from typing import Any

from pypdf import PdfReader


def load_pdf(file_path: str) -> list[dict[str, Any]]:
    """Extract non-empty PDF pages, retaining each page's one-based number."""
    reader = PdfReader(file_path)
    pages_data: list[dict[str, Any]] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages_data.append({
                "page_number": page_num,
                "text": text.strip()
            })

    return pages_data


if __name__ == "__main__":
    data = load_pdf("data/documents/DBMS_Short_Notes_2_Pages.pdf")
    print(f"Extracted {len(data)} pages from PDF.")