import os
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from Backend.rag.embeddings import EMBEDDING_DIMENSION
from Backend.rag.errors import RAGConfigurationError, RAGServiceError

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

class PineconeVectorStore:
    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RAGConfigurationError("PINECONE_API_KEY is not configured.")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "academic-rag-index")
        try:
            self.pc = Pinecone(api_key=api_key)
            listed_indexes = self.pc.list_indexes()
            if hasattr(listed_indexes, "names"):
                index_names = listed_indexes.names()
            else:
                index_names = [
                    index.get("name") if isinstance(index, dict) else index.name
                    for index in listed_indexes
                ]

            if self.index_name not in index_names:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                    timeout=120,
                )
            else:
                description = self.pc.describe_index(self.index_name)
                dimension = (
                    description.get("dimension")
                    if isinstance(description, dict)
                    else description.dimension
                )
                if dimension != EMBEDDING_DIMENSION:
                    raise RAGConfigurationError(
                        f"Pinecone index '{self.index_name}' has dimension {dimension}; "
                        f"the embedding model requires {EMBEDDING_DIMENSION}."
                    )

            self.index = self.pc.Index(self.index_name)
        except RAGConfigurationError:
            raise
        except Exception as exc:
            raise RAGServiceError("Pinecone could not be initialized or reached.") from exc

    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """Upsert chunk vectors and citation metadata in batches."""
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")

        vectors_to_upsert = []
        for chunk, embedding in zip(chunks, embeddings):
            if len(embedding) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Each embedding must contain {EMBEDDING_DIMENSION} values."
                )
            if not chunk.get("id") or not chunk.get("text") or not chunk.get("metadata"):
                raise ValueError("Each chunk must include an id, text, and metadata.")
            metadata = dict(chunk["metadata"])
            metadata["text"] = chunk["text"]
            vectors_to_upsert.append({
                "id": chunk["id"],
                "values": embedding,
                "metadata": metadata,
            })

        try:
            for start in range(0, len(vectors_to_upsert), 100):
                self.index.upsert(vectors=vectors_to_upsert[start:start + 100])
        except Exception as exc:
            raise RAGServiceError("Pinecone could not store the document vectors.") from exc

    def query(self, embedding: list[float], top_k: int):
        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(f"Query embeddings must have {EMBEDDING_DIMENSION} values.")
        try:
            return self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
            )
        except Exception as exc:
            raise RAGServiceError("Pinecone semantic search failed.") from exc