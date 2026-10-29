from sentence_transformers import SentenceTransformer

from Backend.rag.errors import RAGServiceError

EMBEDDING_DIMENSION = 384

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name)
            dimension = self.model.get_embedding_dimension()
        except Exception as exc:
            raise RAGServiceError("The embedding model could not be loaded.") from exc
        if dimension != EMBEDDING_DIMENSION:
            raise RAGServiceError(
                f"The configured embedding model must produce {EMBEDDING_DIMENSION}-dimension vectors."
            )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate 384-dimensional embeddings for document chunks."""
        if not texts:
            return []
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            vectors = embeddings.tolist()
        except Exception as exc:
            raise RAGServiceError("Text embeddings could not be generated.") from exc
        if len(vectors) != len(texts) or any(len(vector) != EMBEDDING_DIMENSION for vector in vectors):
            raise RAGServiceError("The embedding model returned vectors with an invalid shape.")
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Generate a 384-dimensional embedding for a search query."""
        if not query.strip():
            raise ValueError("The query must not be empty.")
        try:
            embedding = self.model.encode(query).tolist()
        except Exception as exc:
            raise RAGServiceError("The query embedding could not be generated.") from exc
        if len(embedding) != EMBEDDING_DIMENSION:
            raise RAGServiceError("The embedding model returned a vector with an invalid dimension.")
        return embedding