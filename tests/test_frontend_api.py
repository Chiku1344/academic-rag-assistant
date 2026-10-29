import os
import unittest
from unittest.mock import Mock, patch

import requests

from Backend.frontend.api_client import APIError, AcademicRAGClient, default_backend_url


class FrontendAPIClientTests(unittest.TestCase):
    def test_default_backend_url_is_configurable(self):
        with patch.dict(os.environ, {"ACADEMIC_RAG_API_URL": "http://localhost:9000/"}):
            self.assertEqual(default_backend_url(), "http://localhost:9000")

    @patch("Backend.frontend.api_client.requests.request")
    def test_question_submission_uses_real_api_contract(self, request):
        response = Mock(ok=True)
        response.json.return_value = {
            "question": "What is normalization?",
            "answer": "It reduces redundancy.",
            "sources": [
                {
                    "source": "dbms.pdf",
                    "page": 4,
                    "text": "Normalization reduces redundancy.",
                    "score": 0.9,
                }
            ],
        }
        request.return_value = response

        result = AcademicRAGClient("http://localhost:8001").ask("What is normalization?")

        self.assertEqual(result["answer"], "It reduces redundancy.")
        self.assertEqual(result["sources"][0]["source"], "dbms.pdf")
        self.assertEqual(result["sources"][0]["page"], 4)
        self.assertEqual(result["sources"][0]["text"], "Normalization reduces redundancy.")
        self.assertEqual(request.call_args.args[:2], ("POST", "http://localhost:8001/ask"))
        self.assertEqual(
            request.call_args.kwargs["json"],
            {"question": "What is normalization?", "top_k": 3},
        )

    @patch("Backend.frontend.api_client.requests.request")
    def test_upload_requires_backend_success_fields(self, request):
        response = Mock(ok=True)
        response.json.return_value = {
            "filename": "dbms.pdf",
            "total_pages": 2,
            "total_chunks": 6,
        }
        request.return_value = response

        result = AcademicRAGClient().upload("dbms.pdf", b"%PDF-1.7")

        self.assertEqual(result["filename"], "dbms.pdf")
        self.assertEqual(result["total_pages"], 2)
        self.assertEqual(result["total_chunks"], 6)
        self.assertEqual(request.call_args.kwargs["files"]["file"][0], "dbms.pdf")

    @patch("Backend.frontend.api_client.requests.request")
    def test_failed_upload_is_not_reported_as_success(self, request):
        response = Mock(ok=False, status_code=503)
        response.json.return_value = {"detail": "PINECONE_API_KEY is not configured."}
        request.return_value = response

        with self.assertRaisesRegex(APIError, "PINECONE_API_KEY is not configured"):
            AcademicRAGClient().upload("dbms.pdf", b"%PDF-1.7")

    @patch("Backend.frontend.api_client.requests.request")
    def test_unavailable_backend_has_actionable_error(self, request):
        request.side_effect = requests.ConnectionError("connection refused")

        with self.assertRaisesRegex(APIError, "make sure the API is running"):
            AcademicRAGClient().health()

    @patch("Backend.frontend.api_client.requests.request")
    def test_history_requires_a_list_response(self, request):
        response = Mock(ok=True)
        response.json.return_value = {"unexpected": True}
        request.return_value = response

        with self.assertRaisesRegex(APIError, "history response was unexpected"):
            AcademicRAGClient().history()


if __name__ == "__main__":
    unittest.main()
