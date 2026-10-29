from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

DATABASE_URL = "sqlite:///./academic_rag.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, index=True)
    answer = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class DocumentLog(Base):
    __tablename__ = "document_logs"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    total_pages = Column(Integer, nullable=True)
    total_chunks = Column(Integer, nullable=True)

Base.metadata.create_all(bind=engine)


def ensure_document_log_metrics(target_engine=engine) -> None:
    """Add nullable counts to existing document logs without changing stored rows."""
    inspector = inspect(target_engine)
    if "document_logs" not in inspector.get_table_names():
        return

    existing_columns = {
        column["name"] for column in inspector.get_columns("document_logs")
    }
    with target_engine.begin() as connection:
        for column_name in ("total_pages", "total_chunks"):
            if column_name not in existing_columns:
                connection.exec_driver_sql(
                    f"ALTER TABLE document_logs ADD COLUMN {column_name} INTEGER"
                )


ensure_document_log_metrics()