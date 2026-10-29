import os
import tempfile
import unittest
import importlib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from Backend.api import documents
from Backend.main import app, get_db
from Backend.rag import chunker, embeddings, generator, loader, prompt, retriever, vector_store
from Backend.rag.errors import RAGConfigurationError, RAGServiceError


class PDFAndChunkingTests(unittest.TestCase):
    def test_pdf_loader_preserves_original_one_based_page_numbers(self):
        pages = [
            SimpleNamespace(extract_text=lambda: "first page"),
            SimpleNamespace(extract_text=lambda: "   "),
            SimpleNamespace(extract_text=lambda: "third page"),
        ]
        with patch.object(loader, "PdfReader", return_value=SimpleNamespace(pages=pages)):
            result = loader.load_pdf("ignored.pdf")

        self.assertEqual(
            result,
            [
                {"page_number": 1, "text": "first page"},
                {"page_number": 3, "text": "third page"},
            ],
        )

    def test_chunker_assigns_source_page_and_unique_ids(self):
        result = chunker.chunk_documents(
            [
                {"page_number": 2, "text": "A " * 80},
                {"page_number": 4, "text": "Page four text."},
            ],
            source_name="lecture.pdf",
            chunk_size=40,
            chunk_overlap=5,
        )

        self.assertGreater(len(result), 2)
        self.assertEqual(len({chunk["id"] for chunk in result}), len(result))
        self.assertEqual({chunk["metadata"]["source"] for chunk in result}, {"lecture.pdf"})
        self.assertEqual({chunk["metadata"]["page"] for chunk in result}, {2, 4})
        self.assertTrue(all(chunk["text"] for chunk in result))

    def test_chunker_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunker.chunk_documents([], "source.pdf", chunk_size=10, chunk_overlap=10)


class EmbeddingAndVectorStoreTests(unittest.TestCase):
    def test_embedding_manager_checks_model_output_dimension(self):
        fake_model = Mock()
        fake_model.get_sentence_embedding_dimension.return_value = 384
        fake_model.encode.return_value.tolist.return_value = [[0.1] * 384]
        with patch.object(embeddings, "SentenceTransformer", return_value=fake_model):
            manager = embeddings.EmbeddingManager()
            vectors = manager.embed_texts(["some academic text"])

        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 384)

    def test_embedding_manager_rejects_incompatible_model(self):
        fake_model = Mock()
        fake_model.get_sentence_embedding_dimension.return_value = 768
        with patch.object(embeddings, "SentenceTransformer", return_value=fake_model):
            with self.assertRaises(RAGServiceError):
                embeddings.EmbeddingManager()

    def test_upsert_preserves_chunk_and_batches_vectors(self):
        store = vector_store.PineconeVectorStore.__new__(vector_store.PineconeVectorStore)
        store.index = Mock()
        chunk = {
            "id": "notes_chunk_1",
            "text": "retrievable text",
            "metadata": {"source": "notes.pdf", "page": 2},
        }

        store.upsert_chunks([chunk], [[0.0] * 384])

        store.index.upsert.assert_called_once()
        vector = store.index.upsert.call_args.kwargs["vectors"][0]
        self.assertEqual(vector["metadata"]["text"], "retrievable text")
        self.assertEqual(vector["metadata"]["page"], 2)
        self.assertNotIn("text", chunk["metadata"])

    def test_upsert_rejects_mismatched_embedding_counts(self):
        store = vector_store.PineconeVectorStore.__new__(vector_store.PineconeVectorStore)
        store.index = Mock()

        with self.assertRaises(ValueError):
            store.upsert_chunks([{"id": "a", "text": "b", "metadata": {}}], [])

    def test_existing_pinecone_index_must_match_embedding_dimension(self):
        fake_pc = Mock()
        fake_pc.list_indexes.return_value.names.return_value = ["academic-rag-index"]
        fake_pc.describe_index.return_value.dimension = 768
        with (
            patch.dict(os.environ, {"PINECONE_API_KEY": "test-key"}),
            patch.object(vector_store, "Pinecone", return_value=fake_pc),
        ):
            with self.assertRaises(RAGConfigurationError):
                vector_store.PineconeVectorStore()
        fake_pc.Index.assert_not_called()

    def test_pinecone_missing_key_fails_before_client_creation(self):
        with (
            patch.dict(os.environ, {"PINECONE_API_KEY": ""}),
            patch.object(vector_store, "Pinecone") as pinecone_client,
        ):
            with self.assertRaises(RAGConfigurationError):
                vector_store.PineconeVectorStore()
        pinecone_client.assert_not_called()


class RetrievalAndGenerationTests(unittest.TestCase):
    def test_retriever_returns_flat_citation_data_and_skips_empty_metadata(self):
        match_with_text = SimpleNamespace(
            score=0.9,
            metadata={"text": "retrieved excerpt", "source": "paper.pdf", "page": 8},
        )
        match_without_text = SimpleNamespace(score=0.2, metadata={"source": "empty.pdf"})
        fake_store = Mock()
        fake_store.query.return_value = SimpleNamespace(
            matches=[match_with_text, match_without_text]
        )
        fake_embedder = Mock()
        fake_embedder.embed_query.return_value = [0.0] * 384

        with (
            patch.object(retriever, "EmbeddingManager", return_value=fake_embedder),
            patch.object(retriever, "PineconeVectorStore", return_value=fake_store),
        ):
            result = retriever.Retriever().search("question")

        self.assertEqual(
            result,
            [{"score": 0.9, "text": "retrieved excerpt", "page": 8, "source": "paper.pdf"}],
        )

    def test_prompt_includes_document_name_and_page(self):
        result = prompt.build_academic_prompt(
            "What is the finding?",
            [{"source": "paper.pdf", "page": 5, "text": "The finding is significant."}],
        )

        self.assertIn("paper.pdf", result)
        self.assertIn("Page 5", result)
        self.assertIn("The finding is significant.", result)

    def test_generator_returns_no_context_message_without_calling_provider(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}), patch.object(
            generator, "Groq"
        ) as groq_client:
            answer = generator.AnswerGenerator().generate_answer("question", [])

        self.assertIn("couldn't find sufficient information", answer)
        groq_client.return_value.chat.completions.create.assert_not_called()

    def test_generator_requires_key_only_when_context_exists(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            instance = generator.AnswerGenerator()
            self.assertIn(
                "couldn't find sufficient information",
                instance.generate_answer("question", []),
            )
            with self.assertRaises(RAGConfigurationError):
                instance.generate_answer(
                    "question",
                    [{"source": "paper.pdf", "page": 1, "text": "context"}],
                )

    def test_generator_returns_provider_answer_with_citation_context(self):
        client = Mock()
        client.chat.completions.create.return_value.choices = [
            SimpleNamespace(message=SimpleNamespace(content="Grounded answer."))
        ]
        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}),
            patch.object(generator, "Groq", return_value=client),
        ):
            answer = generator.AnswerGenerator().generate_answer(
                "What was found?",
                [{"source": "paper.pdf", "page": 6, "text": "The result was positive."}],
            )

        self.assertEqual(answer, "Grounded answer.")
        prompt_text = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("paper.pdf", prompt_text)
        self.assertIn("Page 6", prompt_text)

    def test_all_backend_rag_modules_import_without_provider_startup(self):
        modules = [
            "Backend.rag.ingest",
            "Backend.rag.pdf_processor",
            "Backend.rag.llm_chain",
        ]

        for module_name in modules:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_generator_wraps_provider_failures_without_exposing_provider_details(self):
        client = Mock()
        client.chat.completions.create.side_effect = RuntimeError("private provider response")
        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}),
            patch.object(generator, "Groq", return_value=client),
        ):
            instance = generator.AnswerGenerator()
            with self.assertRaises(RAGServiceError) as raised:
                instance.generate_answer(
                    "question",
                    [{"source": "paper.pdf", "page": 1, "text": "context"}],
                )

        self.assertNotIn("private provider response", str(raised.exception))


class APITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        app.dependency_overrides[get_db] = lambda: Mock()

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_health_and_openapi_docs_are_available(self):
        health_response = self.client.get("/health")
        docs_response = self.client.get("/docs")
        openapi_response = self.client.get("/openapi.json")

        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(health_response.json(), {"status": "ok"})
        self.assertEqual(docs_response.status_code, 200)
        self.assertIn("text/html", docs_response.headers["content-type"])
        self.assertEqual(openapi_response.status_code, 200)

    def test_status_reports_configuration_presence_without_secret_values(self):
        db = Mock()
        db.execute.return_value = 1
        app.dependency_overrides[get_db] = lambda: db
        with patch.dict(
            os.environ,
            {"PINECONE_API_KEY": "test-key", "GROQ_API_KEY": ""},
        ):
            response = self.client.get("/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "database_available": True,
                "pinecone_configured": True,
                "llm_configured": False,
            },
        )
        self.assertNotIn("test-key", response.text)

    def test_history_returns_saved_database_records(self):
        entry = SimpleNamespace(
            id=5,
            question="Saved question",
            answer="Saved answer",
            timestamp=datetime(2026, 10, 29, 12, 0),
        )
        db = Mock()
        db.query.return_value.order_by.return_value.limit.return_value.all.return_value = [entry]
        app.dependency_overrides[get_db] = lambda: db

        response = self.client.get("/history")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["id"], 5)
        self.assertEqual(response.json()[0]["question"], "Saved question")
        self.assertEqual(response.json()[0]["answer"], "Saved answer")
        self.assertEqual(response.json()[0]["timestamp"], "2026-10-29T12:00:00")

    def test_ask_returns_citations_matching_frontend_contract(self):
        fake_retriever = Mock()
        fake_retriever.search.return_value = [
            {
                "source": "paper.pdf",
                "page": 3,
                "text": "retrieved excerpt",
                "score": 0.876543,
            }
        ]
        fake_generator = Mock()
        fake_generator.generate_answer.return_value = "A grounded answer."
        with (
            patch("Backend.main.Retriever", return_value=fake_retriever),
            patch("Backend.main.AnswerGenerator", return_value=fake_generator),
        ):
            response = self.client.post(
                "/ask",
                json={"question": "What is the result?", "top_k": 3},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["sources"],
            [
                {
                    "source": "paper.pdf",
                    "page": 3,
                    "text": "retrieved excerpt",
                    "score": 0.8765,
                }
            ],
        )

    def test_ask_returns_service_unavailable_for_missing_configuration(self):
        with patch(
            "Backend.main.Retriever",
            side_effect=RAGConfigurationError("PINECONE_API_KEY is not configured."),
        ):
            response = self.client.post("/ask", json={"question": "question"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "PINECONE_API_KEY is not configured.")

    def test_upload_validates_pdf_and_ingests_it(self):
        fake_embedder = Mock()
        fake_embedder.embed_texts.return_value = [[0.0] * 384]
        fake_store = Mock()
        page_data = [{"page_number": 1, "text": "PDF text"}]
        chunk_data = [
            {
                "id": "paper.pdf_chunk_1",
                "text": "PDF text",
                "metadata": {"source": "paper.pdf", "page": 1},
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(documents, "UPLOAD_DIR", Path(temp_dir)),
                patch.object(documents, "load_pdf", return_value=page_data),
                patch.object(documents, "chunk_documents", return_value=chunk_data),
                patch.object(documents, "EmbeddingManager", return_value=fake_embedder),
                patch.object(documents, "PineconeVectorStore", return_value=fake_store),
            ):
                response = self.client.post(
                    "/documents/upload",
                    files={"file": ("paper.PDF", b"%PDF-1.7 mocked body", "application/pdf")},
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["total_pages"], 1)
            self.assertEqual(response.json()["total_chunks"], 1)
            self.assertTrue((Path(temp_dir) / "paper.PDF").exists())
            fake_store.upsert_chunks.assert_called_once_with(
                chunk_data, fake_embedder.embed_texts.return_value
            )

    def test_upload_rejects_non_pdf_bytes(self):
        response = self.client.post(
            "/documents/upload",
            files={"file": ("paper.pdf", b"not a pdf", "application/pdf")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "The uploaded file is not a valid PDF.")

    def test_upload_rejects_path_like_filename(self):
        response = self.client.post(
            "/documents/upload",
            files={"file": ("..\\outside.pdf", b"%PDF-1.7", "application/pdf")},
        )

        self.assertEqual(response.status_code, 400)

    def test_upload_uses_a_new_name_when_the_filename_already_exists(self):
        fake_embedder = Mock()
        fake_embedder.embed_texts.return_value = [[0.0] * 384]
        page_data = [{"page_number": 1, "text": "PDF text"}]
        chunk_data = [
            {
                "id": "paper.pdf_chunk_1",
                "text": "PDF text",
                "metadata": {"source": "paper.pdf", "page": 1},
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "paper.pdf").write_bytes(b"existing file")
            with (
                patch.object(documents, "UPLOAD_DIR", Path(temp_dir)),
                patch.object(documents, "load_pdf", return_value=page_data),
                patch.object(documents, "chunk_documents", return_value=chunk_data),
                patch.object(documents, "EmbeddingManager", return_value=fake_embedder),
                patch.object(documents, "PineconeVectorStore"),
            ):
                response = self.client.post(
                    "/documents/upload",
                    files={"file": ("paper.pdf", b"%PDF-1.7 mocked body", "application/pdf")},
                )

            self.assertEqual(response.status_code, 200)
            stored_name = response.json()["filename"]
            self.assertRegex(stored_name, r"^paper_[0-9a-f]{8}\.pdf$")
            self.assertEqual((Path(temp_dir) / "paper.pdf").read_bytes(), b"existing file")
            self.assertTrue((Path(temp_dir) / stored_name).exists())


if __name__ == "__main__":
    unittest.main()
