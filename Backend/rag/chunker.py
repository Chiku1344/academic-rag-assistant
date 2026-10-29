from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    pages_data: list[dict[str, Any]],
    source_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict[str, Any]]:
    """Split page text into chunks while preserving source and page metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks: list[dict[str, Any]] = []
    chunk_id = 0

    for page in pages_data:
        text = page.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        page_number = page.get("page_number")
        if not isinstance(page_number, int) or page_number < 1:
            raise ValueError("Each page must have a positive one-based page_number.")

        splits = text_splitter.split_text(text.strip())
        for split_text in splits:
            chunk_id += 1
            chunks.append({
                "id": f"{source_name}_chunk_{chunk_id}",
                "text": split_text,
                "metadata": {
                    "source": source_name,
                    "page": page_number,
                }
            })

    return chunks