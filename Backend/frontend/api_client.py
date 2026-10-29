import os
from typing import Any

import requests


class APIError(RuntimeError):
    """A safe, user-facing error returned while communicating with FastAPI."""


def default_backend_url() -> str:
    return os.getenv("ACADEMIC_RAG_API_URL", "http://127.0.0.1:8001").rstrip("/")


class AcademicRAGClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or default_backend_url()).rstrip("/")

    def _request(self, method: str, path: str, *, timeout: int, **kwargs: Any) -> Any:
        try:
            response = requests.request(
                method,
                f"{self.base_url}{path}",
                timeout=timeout,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise APIError(
                "Could not reach the FastAPI service. Check the backend URL and make sure the API is running."
            ) from exc

        try:
            payload = response.json()
        except ValueError:
            payload = None

        if not response.ok:
            message = payload.get("detail") if isinstance(payload, dict) else None
            raise APIError(
                str(message) if message else f"The API returned HTTP {response.status_code}."
            )
        if payload is None:
            raise APIError("The API returned an invalid response.")
        return payload

    def health(self) -> dict[str, Any]:
        payload = self._request("GET", "/health", timeout=5)
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise APIError("The API health response was unexpected.")
        return payload

    def status(self) -> dict[str, Any]:
        payload = self._request("GET", "/status", timeout=5)
        if not isinstance(payload, dict):
            raise APIError("The API status response was unexpected.")
        return payload

    def history(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/history", timeout=15)
        if not isinstance(payload, list):
            raise APIError("The query history response was unexpected.")
        return payload

    def ask(self, question: str, top_k: int = 3) -> dict[str, Any]:
        payload = self._request(
            "POST",
            "/ask",
            timeout=180,
            json={"question": question, "top_k": top_k},
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("answer"), str):
            raise APIError("The answer response was missing its answer.")
        sources = payload.get("sources", [])
        if not isinstance(sources, list) or any(
            not isinstance(source, dict) for source in sources
        ):
            raise APIError("The answer response contained invalid source citations.")
        payload["sources"] = sources
        return payload

    def upload(self, filename: str, content: bytes) -> dict[str, Any]:
        payload = self._request(
            "POST",
            "/documents/upload",
            timeout=600,
            files={"file": (filename, content, "application/pdf")},
        )
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("filename"), str)
            or not isinstance(payload.get("total_pages"), int)
            or not isinstance(payload.get("total_chunks"), int)
        ):
            raise APIError("The upload response did not confirm successful document processing.")
        return payload
