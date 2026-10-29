from pathlib import Path

from .ingest import run_ingestion


def process_and_index_pdf(pdf_path: str) -> int:
    """Ingest a PDF through the same page-aware pipeline as the upload API."""
    return run_ingestion(pdf_path, source_name=Path(pdf_path).name)
