class RAGConfigurationError(RuntimeError):
    """Raised when a required provider configuration is missing or invalid."""


class RAGServiceError(RuntimeError):
    """Raised when an external RAG provider cannot complete a request."""
