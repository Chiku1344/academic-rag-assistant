import os
from typing import Any

import requests

DEFAULT_BACKEND_URL = "http://127.0.0.1:8001"


class APIError(RuntimeError):
    """A safe, user-presentable error returned by or connecting to FastAPI."""


class AcademicRAGAPI:
    def __init__(
        self,
        base_url: str | None = None,
        session: requests.Session | None = None,
    ):
        configured_url = base_url or os.getenv("ACADEMIC_RAG_API_URL", DEFAULT_BACKEND_URL)
        self.base_url = configured_url.rstrip("/")
        self.session = session or requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                timeout=kwargs.pop("timeout", 10),
                **kwargs,
            )
        except requests.RequestException as exc:
            raise APIError(
                "FastAPI is unavailable. Start the backend and check its configured URL."
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise APIError("The backend returned an invalid response.") from exc

        if not response.ok:
            detail = payload.get("detail") if isinstance(payload, dict) else None
            raise APIError(str(detail or f"Backend request failed ({response.status_code})."))
        if not isinstance(payload, dict):
            raise APIError("The backend returned an unexpected response format.")
        return payload

    def health(self) -> dict[str, Any]:
        payload = self._request("GET", "/health", timeout=2)
        if payload.get("status") != "ok":
            raise APIError("The backend health check returned an unexpected status.")
        return payload

    def get_documents(self) -> dict[str, Any]:
        payload = self._request("GET", "/documents")
        documents = payload.get("documents")
        total = payload.get("total_documents")
        if not isinstance(documents, list) or not isinstance(total, int):
            raise APIError("The document list response has an unexpected format.")
        return payload

    def get_history(self, limit: int = 50) -> dict[str, Any]:
        payload = self._request("GET", "/history", params={"limit": limit})
        if not isinstance(payload.get("history"), list):
            raise APIError("The history response has an unexpected format.")
        return payload

    def get_status(self) -> dict[str, Any]:
        payload = self._request("GET", "/status")
        expected = ("pinecone_configured", "groq_configured")
        if any(not isinstance(payload.get(key), bool) for key in expected):
            raise APIError("The configuration status response has an unexpected format.")
        return payload

    def ask(self, question: str, top_k: int = 3) -> dict[str, Any]:
        payload = self._request(
            "POST",
            "/ask",
            json={"question": question, "top_k": top_k},
            timeout=180,
        )
        if not isinstance(payload.get("answer"), str) or not isinstance(
            payload.get("sources"), list
        ):
            raise APIError("The answer response has an unexpected format.")
        return payload

    def upload_pdf(self, filename: str, content: bytes) -> dict[str, Any]:
        payload = self._request(
            "POST",
            "/documents/upload",
            files={"file": (filename, content, "application/pdf")},
            timeout=600,
        )
        if not all(
            isinstance(payload.get(key), expected_type)
            for key, expected_type in (
                ("filename", str),
                ("total_pages", int),
                ("total_chunks", int),
            )
        ):
            raise APIError("The upload response has an unexpected format.")
        return payload