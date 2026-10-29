import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from Backend.api.documents import router as documents_router
from Backend.database.models import QueryLog, SessionLocal
from Backend.rag.errors import RAGConfigurationError, RAGServiceError
from Backend.rag.generator import AnswerGenerator
from Backend.rag.retriever import Retriever

app = FastAPI(title="Academic RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)


@app.get("/")
def root():
    return {"message": "Academic RAG API is live!"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


@app.get("/status", tags=["Health"])
def status(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        database_available = True
    except Exception:
        db.rollback()
        database_available = False

    return {
        "database_available": database_available,
        "pinecone_configured": bool(os.getenv("PINECONE_API_KEY")),
        "llm_configured": bool(os.getenv("GROQ_API_KEY")),
    }


@app.get("/history", tags=["History"])
def query_history(db: Session = Depends(get_db)):
    entries = (
        db.query(QueryLog)
        .order_by(QueryLog.timestamp.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "id": entry.id,
            "question": entry.question,
            "answer": entry.answer,
            "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
        }
        for entry in entries
    ]


@app.post("/ask")
def ask_question(request: QuestionRequest, db: Session = Depends(get_db)):
    try:
        docs = Retriever().search(request.question, top_k=request.top_k)
        answer = AnswerGenerator().generate_answer(request.question, docs)

        sources = [
            {
                "source": doc["source"],
                "page": doc["page"],
                "text": doc["text"],
                "score": round(doc["score"], 4) if doc["score"] is not None else None,
            }
            for doc in docs
        ]

        db.add(QueryLog(question=request.question, answer=answer))
        db.commit()
        return {
            "question": request.question,
            "answer": answer,
            "sources": sources,
        }
    except (RAGConfigurationError, RAGServiceError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to process the question.") from exc
