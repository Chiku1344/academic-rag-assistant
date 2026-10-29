from pathlib import Path, PureWindowsPath
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from Backend.database.models import DocumentLog, SessionLocal
from Backend.rag.loader import load_pdf
from Backend.rag.chunker import chunk_documents
from Backend.rag.embeddings import EmbeddingManager
from Backend.rag.errors import RAGConfigurationError, RAGServiceError
from Backend.rag.vector_store import PineconeVectorStore

router = APIRouter(prefix="/documents", tags=["Documents"])

UPLOAD_DIR = Path(__file__).resolve().parents[2] / "data" / "documents"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    filename = file.filename
    if (
        not filename
        or Path(filename).name != filename
        or PureWindowsPath(filename).name != filename
        or any(character in filename for character in '<>:"/\\|?*')
        or filename.endswith((" ", "."))
        or any(ord(character) < 32 for character in filename)
    ):
        raise HTTPException(status_code=400, detail="A valid PDF filename is required.")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The PDF exceeds the 25 MB upload limit.")
    if b"%PDF-" not in content[:1024]:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")

    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path = UPLOAD_DIR / f"{Path(filename).stem}_{uuid4().hex[:8]}{Path(filename).suffix}"
        file_path.write_bytes(content)

        try:
            pages = load_pdf(str(file_path))
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Unable to read a valid PDF.") from exc
        if not pages:
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF.")

        source_name = file_path.name
        chunks = chunk_documents(pages, source_name=source_name)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF.")

        embedder = EmbeddingManager()
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.embed_texts(texts)

        vector_store = PineconeVectorStore()
        vector_store.upsert_chunks(chunks, embeddings)
        db.add(
            DocumentLog(
                filename=file_path.name,
                total_pages=len(pages),
                total_chunks=len(chunks),
            )
        )
        db.commit()

        return {
            "message": "File uploaded and ingested successfully!",
            "filename": file_path.name,
            "total_pages": len(pages),
            "total_chunks": len(chunks)
        }
    except HTTPException:
        raise
    except (RAGConfigurationError, RAGServiceError) as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to process and index the PDF.") from exc