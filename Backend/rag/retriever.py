from .embeddings import EmbeddingManager
from .vector_store import PineconeVectorStore


class Retriever:
    def __init__(self):
        self.vector_store = PineconeVectorStore()
        self.embedder = EmbeddingManager()

    def search(self, query: str, top_k: int = 3):
        """Retrieves top-k relevant document chunks for a question."""
        query_vector = self.embedder.embed_query(query)

        results = self.vector_store.query(query_vector, top_k=top_k)
        retrieved_docs = []
        matches = results.get("matches", []) if isinstance(results, dict) else results.matches
        for match in matches:
            metadata = match.get("metadata", {}) if isinstance(match, dict) else (match.metadata or {})
            score = match.get("score") if isinstance(match, dict) else match.score
            text = metadata.get("text")
            if not text:
                continue
            retrieved_docs.append({
                "score": score,
                "text": text,
                "page": metadata.get("page"),
                "source": metadata.get("source", "Unknown document"),
            })

        return retrieved_docs