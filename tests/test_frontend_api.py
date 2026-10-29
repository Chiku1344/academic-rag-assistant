import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import requests
from streamlit.testing.v1 import AppTest

from Backend.frontend.api_client import (
    APIError,
    AcademicRAGAPI,
    DEFAULT_BACKEND_URL,
)


class AcademicRAGAPITests(unittest.TestCase):
    def setUp(self):
        self.session = Mock()
        self.response = Mock()
        self.response.ok = True
        self.response.status_code = 200
        self.session.request.return_value = self.response
        self.api = AcademicRAGAPI(
            "http://localhost:8001/",
            session=self.session,
        )

    def test_default_backend_url_is_local_fastapi(self):
        self.assertEqual(DEFAULT_BACKEND_URL, "http://127.0.0.1:8001")

    def test_ask_sends_backend_schema_and_accepts_citations(self):
        self.response.json.return_value = {
            "question": "What is normalization?",
            "answer": "It reduces redundancy.",
            "sources": [
                {
                    "source": "dbms.pdf",
                    "page": 4,
                    "text": "Normalization reduces data redundancy.",
                    "score": 0.9,
                }
            ],
        }

        result = self.api.ask("What is normalization?", top_k=3)

        self.assertEqual(result["sources"][0]["page"], 4)
        request = self.session.request.call_args
        self.assertEqual(request.args[:2], ("POST", "http://localhost:8001/ask"))
        self.assertEqual(
            request.kwargs["json"],
            {"question": "What is normalization?", "top_k": 3},
        )

    def test_upload_sends_pdf_and_accepts_confirmed_ingestion_response(self):
        self.response.json.return_value = {
            "message": "File uploaded and ingested successfully!",
            "filename": "dbms.pdf",
            "total_pages": 8,
            "total_chunks": 23,
        }

        result = self.api.upload_pdf("dbms.pdf", b"%PDF-1.7 data")

        self.assertEqual(result["total_pages"], 8)
        self.assertEqual(result["total_chunks"], 23)
        request = self.session.request.call_args
        self.assertEqual(request.args[:2], ("POST", "http://localhost:8001/documents/upload"))
        self.assertEqual(
            request.kwargs["files"]["file"],
            ("dbms.pdf", b"%PDF-1.7 data", "application/pdf"),
        )

    def test_document_library_and_history_use_backend_read_routes(self):
        self.response.json.return_value = {
            "total_documents": 1,
            "documents": [{"filename": "dbms.pdf", "status": "indexed"}],
        }
        documents = self.api.get_documents()
        self.assertEqual(documents["total_documents"], 1)
        self.assertEqual(
            self.session.request.call_args.args[:2],
            ("GET", "http://localhost:8001/documents"),
        )

        self.response.json.return_value = {
            "history": [{"question": "Q", "answer": "A", "timestamp": None}]
        }
        history = self.api.get_history(limit=15)
        self.assertEqual(len(history["history"]), 1)
        self.assertEqual(
            self.session.request.call_args.kwargs["params"],
            {"limit": 15},
        )

    def test_backend_http_error_uses_backend_detail(self):
        self.response.ok = False
        self.response.json.return_value = {"detail": "PINECONE_API_KEY is not configured."}

        with self.assertRaisesRegex(APIError, "PINECONE_API_KEY is not configured"):
            self.api.health()

    def test_unavailable_backend_returns_helpful_safe_error(self):
        self.session.request.side_effect = requests.ConnectionError("internal detail")

        with self.assertRaisesRegex(APIError, "FastAPI is unavailable"):
            self.api.health()

    def test_configuration_status_requires_booleans_only(self):
        self.response.json.return_value = {
            "pinecone_configured": True,
            "groq_configured": False,
        }

        self.assertEqual(
            self.api.get_status(),
            {"pinecone_configured": True, "groq_configured": False},
        )


class StreamlitWorkflowTests(unittest.TestCase):
    app_path = str(
        Path(__file__).resolve().parents[1]
        / "Backend"
        / "frontend"
        / "app.py"
    )

    def test_question_submission_renders_mock_answer_and_source(self):
        response = {
            "question": "Explain normalization",
            "answer": "It reduces redundancy.",
            "sources": [
                {
                    "source": "dbms.pdf",
                    "page": 2,
                    "text": "Normalization reduces redundancy.",
                    "score": 0.92,
                }
            ],
        }
        with (
            patch.dict(os.environ, {"ACADEMIC_RAG_API_URL": "http://127.0.0.1:9"}),
            patch.object(AcademicRAGAPI, "health", return_value={"status": "ok"}),
            patch.object(
                AcademicRAGAPI,
                "get_documents",
                return_value={"total_documents": 0, "documents": []},
            ),
            patch.object(AcademicRAGAPI, "ask", return_value=response) as ask,
        ):
            app = AppTest.from_file(self.app_path, default_timeout=20).run()
            app.text_area[0].set_value("Explain normalization")
            app.button[-1].click().run()

        self.assertEqual(ask.call_args.args, ("Explain normalization",))
        self.assertFalse(app.exception)
        rendered_markdown = "\n".join(str(element.value) for element in app.markdown)
        self.assertIn("It reduces redundancy.", rendered_markdown)
        self.assertIn("dbms.pdf", rendered_markdown)

    def test_all_navigation_screens_render_without_fake_backend_data(self):
        with (
            patch.dict(os.environ, {"ACADEMIC_RAG_API_URL": "http://127.0.0.1:9"}),
            patch.object(AcademicRAGAPI, "health", return_value={"status": "ok"}),
            patch.object(
                AcademicRAGAPI,
                "get_documents",
                return_value={"total_documents": 0, "documents": []},
            ),
            patch.object(AcademicRAGAPI, "get_history", return_value={"history": []}),
            patch.object(
                AcademicRAGAPI,
                "get_status",
                return_value={
                    "pinecone_configured": False,
                    "groq_configured": False,
                },
            ),
        ):
            app = AppTest.from_file(self.app_path, default_timeout=20).run()
            self.assertFalse(app.exception)
            for page in ("My Documents", "History", "Settings"):
                app.radio[0].set_value(page).run()
                self.assertFalse(app.exception, page)
                self.assertEqual(app.title[0].value, page)

    def test_offline_backend_shows_warning_without_app_exception(self):
        with patch.dict(
            os.environ,
            {"ACADEMIC_RAG_API_URL": "http://127.0.0.1:9"},
        ):
            app = AppTest.from_file(self.app_path, default_timeout=20).run()

        self.assertFalse(app.exception)
        self.assertTrue(
            any("API unavailable" in str(item.value) for item in app.warning)
        )


if __name__ == "__main__":
    unittest.main()